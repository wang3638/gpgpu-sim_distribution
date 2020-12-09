// CAWS: Criticality Aware Warp Scheduler

#include "shader.h"

static bool cmp_warps_by_criticality(shd_warp_t* lhs, shd_warp_t* rhs)
{
  assert(lhs != NULL && rhs != NULL);
  if (lhs->done_exit() || lhs->waiting()) {
    return false;
  }
  if (rhs->done_exit() || rhs->waiting()) {
    return true;
  }
  if (lhs->get_cpl() == rhs->get_cpl()) {
    return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
  }
  return lhs->get_cpl() > rhs->get_cpl();
}

// CAWS
caws_scheduler::caws_scheduler(
    shader_core_stats *stats, shader_core_ctx *shader,
    Scoreboard *scoreboard, simt_stack **simt,
    std::vector<shd_warp_t *> *warp, register_set *sp_out,
    register_set *dp_out, register_set *sfu_out,
    register_set *int_out, register_set *tensor_core_out,
    std::vector<register_set *> &spec_cores_out,
    register_set *mem_out, int id)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out,
                     mem_out, id),
      m_count(0), m_flag(0) {}

void caws_scheduler::order_warps()
{
  assert(m_supervised_warps.size() > 0);
  order_by_priority(m_next_cycle_prioritized_warps,
                    m_supervised_warps,
                    m_last_supervised_issued,
                    m_supervised_warps.size());
}

void caws_scheduler::order_by_priority(
    std::vector<shd_warp_t *>& result_list,
    const std::vector<shd_warp_t *>& input_list,
    const std::vector<shd_warp_t*>::const_iterator& last_issued_from_input,
    unsigned num_warps_to_add)
{
  assert(num_warps_to_add <= input_list.size());
  result_list.clear();
  std::vector<shd_warp_t *> temp = input_list;
  sort_warps(temp);
  for (unsigned i = 0; i < num_warps_to_add; ++i) {
    result_list.push_back(temp[i]);
  }
}

void caws_scheduler::sort_warps(std::vector<shd_warp_t*>& temp)
{
  for (unsigned i = 0; i < temp.size() - 1; ++i) {
    unsigned jMax = i;
    for (unsigned j = i + 1; j < temp.size(); ++j) {
      if (cmp_warps_by_criticality(temp[j], temp[jMax])) {
        jMax = j;
      }
    }
    if (jMax != i) {
      shd_warp_t *tmp = temp[i];
      temp[i] = temp[jMax];
      temp[jMax] = tmp;
    }
  }
}
// End CAWS

// GCAWS
gcaws_scheduler::gcaws_scheduler(
    shader_core_stats *stats, shader_core_ctx *shader,
    Scoreboard *scoreboard, simt_stack **simt,
    std::vector<shd_warp_t *> *warp, register_set *sp_out,
    register_set *dp_out, register_set *sfu_out,
    register_set *int_out, register_set *tensor_core_out,
    std::vector<register_set *> &spec_cores_out,
    register_set *mem_out, int id)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out,
                     mem_out, id),
      m_count(0), m_flag(0) {}

void gcaws_scheduler::order_warps()
{
  assert(m_supervised_warps.size() > 0);
  order_by_priority(m_next_cycle_prioritized_warps,
                    m_supervised_warps,
                    m_last_supervised_issued,
                    m_supervised_warps.size() );
}

void gcaws_scheduler::order_by_priority(
    std::vector<shd_warp_t*>& result_list,
    const std::vector<shd_warp_t*>& input_list,
    const std::vector<shd_warp_t*>::const_iterator& last_issued_from_input,
    unsigned num_warps_to_add)
{
  assert(num_warps_to_add <= input_list.size());
  result_list.clear();
  std::vector<shd_warp_t*> temp = input_list;
  sort_warps(temp);

  // Greedy.
  if (last_issued_from_input == input_list.end()) {
    for (unsigned i = 0; i < num_warps_to_add; ++i) {
      result_list.push_back(temp[i]);
    }
  } else {
    shd_warp_t *greedy_value = *last_issued_from_input;
    result_list.push_back(greedy_value);
    for (unsigned i = 0; i < num_warps_to_add; ++i) {
      if (temp[i] != greedy_value) {
        result_list.push_back(temp[i]);
      }
    }
  }
}

void gcaws_scheduler::sort_warps(std::vector<shd_warp_t*>& temp) {
  for (unsigned i = 0; i < temp.size()-1; ++i) {
    unsigned jMax = i;
    for (unsigned j = i+1; j < temp.size(); ++j) {
      if (cmp_warps_by_criticality(temp[j], temp[jMax])) {
        jMax = j;
      }
    }
    if (jMax != i) {
      shd_warp_t* tmp = temp[i];
      temp[i] = temp[jMax];
      temp[jMax] = tmp;
    }
  }
}
// End GCAWS
