// CPL: Criticality Prediction Logic

#include "gpu-sim.h"
#include "shader.h"
#include "../cuda-sim/ptx_sim.h"

#include <float.h>
#include <string.h>
#include <algorithm>

// gpgpu_sim
void gpgpu_sim::print_cpl_accuracy() const
{
  m_shader_stats->print_cpl_accuracy(stdout);
}
// End gpgpu_sim

// shader_core_stats
void shader_core_stats::cpl_launch_kernel(unsigned kid, unsigned total_cta,
                                          unsigned num_warps_per_cta)
{
  for (unsigned i = cpl_actual_vec.size(); i < kid; ++i) {
    int **cpl_actual = (int **)calloc(total_cta + 1, sizeof(int *));
    cpl_actual[0] = (int*)calloc(2, sizeof(int));
    cpl_actual[0][0] = total_cta;
    cpl_actual[0][1] = num_warps_per_cta;
    for (unsigned j = 1; j <= total_cta; ++j) {
      cpl_actual[j] = (int *)calloc(num_warps_per_cta, sizeof(int));
    }
    cpl_actual_vec.push_back(cpl_actual);
  }
  assert(cpl_actual_vec[kid - 1] != NULL);
  cpl_num_launched_kernels++;
}

void shader_core_stats::print_cpl_accuracy(FILE* fp) const
{
  fprintf(fp, "[wsy]: number of total cpl samples = %d\n",
          cpl_total_cpl_for_accuracy);
  fprintf(fp, "[wsy]: number of accurace cpl samples = %d\n",
          cpl_accurate_cpl_for_accuracy);
  fprintf(fp, "[wsy]: cpl accuracy = %.2f%%\n",
          100.0 * cpl_accurate_cpl_for_accuracy /
              cpl_total_cpl_for_accuracy);
}
// End shader_core_stats

// shader_core_config
void shader_core_config::cawa_reg_options(class OptionParser * opp)
{
  option_parser_register(
      opp, "-enable_cacp_l1_cache", OPT_BOOL, &enable_cacp_l1_cache,
      "Enable CACP L1 cache", "0");
}
// End shader_core_config

// shader_core_ctx
void shader_core_ctx::cpl_get_start_end_warp_id(
    unsigned* start_warp_id, unsigned* end_warp_id, unsigned cta_num) const{
  unsigned cta_size = m_kernel->threads_per_cta();
  unsigned padded_cta_size = cta_size;
  if (cta_size%m_config->warp_size) {
    padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);
  }
  assert(padded_cta_size % m_config->warp_size == 0);
  *start_warp_id = cta_num * padded_cta_size / m_config->warp_size;
  *end_warp_id = *start_warp_id + padded_cta_size / m_config->warp_size;
}

std::vector<float> shader_core_ctx::get_current_cpl_counters() const
{
  std::vector<float> ret;
  for (unsigned i = 0; i < m_warp.size(); ++i) {
    ret.push_back(m_warp[i]->get_cpl());
  }
  return ret;
}

void shader_core_ctx::print_cpl_counters(unsigned start_id,
                                         unsigned end_id) const
{
  printf("[wsy] CPL counters for warp %d - warp %d:\n", start_id, end_id);
  std::vector<float> counters = get_current_cpl_counters();
  for (unsigned i = start_id; i < end_id; ++i) {
    printf("W%d: %.2f, ", i, counters[i]);
  }
  printf("\n");
}

void shader_core_ctx::calc_shader_cpl_accuracy() const
{
  if (m_kernel == NULL)
    return;
  for (unsigned i = 0; i < kernel_max_cta_per_shader; ++i) {
    if (m_cta_status[i] == 0)
      break;
    // Find critical warp of the thread block
    unsigned start_warp_id, end_warp_id;
    cpl_get_start_end_warp_id(&start_warp_id, &end_warp_id, i);
    assert(start_warp_id < m_warp.size() && end_warp_id <= m_warp.size());
    unsigned crit_warp = start_warp_id;

    // Check if its actual counter is larger than 50% of the other warps
    unsigned num_larger = 0;
    for (unsigned j = start_warp_id; j < end_warp_id; ++j) {
      if (m_warp[crit_warp]->get_cpl() >= m_warp[j]->get_cpl()) {
        num_larger++;
      }
    }
    if (num_larger >= (end_warp_id-start_warp_id) / 2) {
      m_stats->cpl_accurate_cpl_for_accuracy++;
    }

    m_stats->cpl_total_cpl_for_accuracy++;
  }
}

void shader_core_ctx::calc_shader_cpl(unsigned cycle)
{
  for (unsigned i = 0; i < m_warp.size(); ++i) {
    m_warp[i]->calc_warp_cpl(cycle);
  }

  calc_shader_cpl_accuracy();
}

static unsigned cpl_find_next_pc(unsigned cur_pc, addr_vector_t vec)
{
  assert(vec.size() > 0);
  unsigned min = vec[0];
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] < min)
      min = vec[i];
  }
  unsigned max = vec[0];
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] > max)
      max = vec[i];
  }
  if (min < cur_pc && max > cur_pc) {
    min = max;
    for (unsigned i = 0; i < vec.size(); ++i) {
      if (vec[i] < min && vec[i] > cur_pc)
	min = vec[i];
    }
  }
  return min;
}

address_type shader_core_ctx::calc_npc_per_warp(unsigned warp_id)
{
  unsigned wtid = warp_id * m_warp_size;
  addr_vector_t next_pc;
  for (unsigned i = 0; i < m_warp_size; ++i) {
    if( !ptx_thread_done(wtid+i) ) {
      next_pc.push_back( m_thread[wtid+i]->get_pc() );
    }
  }
  if (next_pc.size() > 0) {
    return cpl_find_next_pc(m_warp[warp_id]->get_pc(), next_pc);
  }
  else{
    return (address_type) -1;
  }
}
// End shader_core_ctx

// shd_warp_t
float shd_warp_t::get_cpl() const
{
  assert(m_shader);
  return cpl_actual;
}

void shd_warp_t::cpl_warp_enter(unsigned cycle, unsigned ninst)
{
  cpl_nInst = ninst;
  cpl_warp_entered_cycle = cycle;
}

void shd_warp_t::cpl_warp_issue(unsigned cycle, address_type npc,
                                unsigned isize)
{
  // Save stall info.
  assert(cycle > cpl_last_schedule_cycle);
  cpl_nStall += (cpl_last_schedule_cycle != 0) ?
                (cycle - cpl_last_schedule_cycle) : 0;
  cpl_last_schedule_cycle = cycle;
  // Calculate nInst.
  if (npc != (address_type) -1) {
    if (npc < m_next_pc) {
      cpl_nInst += (m_next_pc - npc) / isize + 1;
    } else {
      cpl_nInst -= (npc - m_next_pc) / isize - 1;
    }
  }
}

void shd_warp_t::cpl_warp_complete()
{
  cpl_num_completed_inst++;
  cpl_nInst--;
}

void shd_warp_t::calc_warp_cpl(unsigned cycle)
{
  float cpi = 1.0 * (cycle - cpl_warp_entered_cycle) / cpl_num_completed_inst;
  cpl_actual = cpi * cpl_nInst;
  cpl_actual += 1.0 * cpl_nStall;
}

void shd_warp_t::cpl_warp_exit(unsigned cycle)
{
  cpl_warp_completed_cycle = cycle;
}
// End shd_warp_t
