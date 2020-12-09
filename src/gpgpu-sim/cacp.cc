// CACP: Criticality-Aware Cache Prioritization

#include "shader.h"
#include "gpu-cache.h"
#include "stat-tool.h"
#define CRITICAL_PERCENTAGE 0.5

void simt_core_cluster::print_cacp_stats() const
{
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    printf("[wsy] Cluster %u:\n", i);
    m_core[i]->print_cacp_stats();
  }
}

void ldst_unit::print_cacp_stats() const
{
  m_L1D->print_cacp_stats();
}

void shader_core_ctx::calc_warp_criticality()
{
  float max = -1.0 * 0xFFFFFFFF, min = 1.0 * 0xFFFFFFF;
  for (unsigned i = 0; i < m_warp.size(); ++i) {
    if (max < m_warp[i]->get_cpl()) {
      max = m_warp[i]->get_cpl();
    }
    if (min > m_warp[i]->get_cpl()) {
      min = m_warp[i]->get_cpl();
    }
  }
  for (unsigned i = 0; i < m_warp.size(); ++i)
    if ((m_warp[i]->get_cpl() - min) / (max - min) >
            CRITICAL_PERCENTAGE) {
      m_warp[i]->cpl_is_critical = true;
    } else {
      m_warp[i]->cpl_is_critical = false;
    }
}

bool shader_core_ctx::get_warp_criticality(unsigned warp_id) {
  calc_warp_criticality();
  return m_warp[warp_id]->cpl_is_critical;
}

void shader_core_ctx::print_cacp_stats() const {
  printf("[wsy] Core %d:\n", m_sid);
  m_ldst_unit->print_cacp_stats();
}

enum cache_request_status
cacp_l1_cache::access(new_addr_type addr, mem_fetch *mf, unsigned time,
                      std::list<cache_event> &events )
{
  assert(mf->get_data_size() <= m_config.get_line_sz());
  bool wr = mf->get_is_write();
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  bool zero_reuse = false, critical_eviction = false,
       correct_prediction = false;
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, true);

  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(
      mf->get_access_type(),
      m_stats.select_stats_status(probe_status, access_status));
  m_cacp_stats->record_stats(
      probe_status, mf->is_critical(),
      static_cast<cacp_tag_array *>(m_tag_array)->is_correct);
  return access_status;
}

void cacp_l1_cache::print_cacp_stats() const
{
  m_cacp_stats->print_stats();
}

void cacp_cache_stats::record_stats(
    enum cache_request_status status, bool cpl_is_critical, bool correct)
{
  // Record critical accesses.
  if (cpl_is_critical) {
    if (status == HIT) {
      m_total_critical_hit++;
    }
    m_total_critical_access++;
  }

  // Record total accesses.
  if (status == HIT) {
    m_total_hit++;
  }
  m_total_access++;

  // Record CCBP accuracy.
  if (correct) {
    m_ccbp_correct++;
  }
}

void cacp_cache_stats::print_stats() const
{
  printf("[wsy] Critical Hit: %u, Critical Access: %u, Critical Hit Rate: %.2f%%\n",
         m_total_critical_hit, m_total_critical_access,
         100.0 * m_total_critical_hit / m_total_critical_access);
  printf("[wsy] Total Hit: %u, Total Access: %u, Total Hit Rate: %.2f%%\n",
         m_total_hit, m_total_access,
         100.0 * m_total_hit / m_total_access);
  printf("[wsy] CCBP correct: %u, Total CCBP Access: %u, CCBP accuracy: %.2f%%\n",
         m_ccbp_correct, m_total_access,
         100.0 * m_ccbp_correct / m_total_access);
}

enum cache_request_status cacp_tag_array::probe(
    new_addr_type addr, unsigned &idx, mem_access_sector_mask_t mask,
    bool probe_mode, mem_fetch *mf) {
  // The signature is formed by xor-ing the lower 8 bits of an instruction
  // and the memory address.
  unsigned signature = (addr & 256) ^ (mf->get_tpc() & 256);
  unsigned set_index = m_config.set_index(addr);
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;

  for (unsigned way = 0; way < m_config.m_assoc; ++way) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    // Update CACP predictor on hit.
    if (line->m_tag == tag) {
      if (line->get_status(mask) == RESERVED) {
        idx = index;
        this->cache_hit(idx, mf->is_critical());
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        this->cache_hit(idx, mf->is_critical());
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if (line->is_readable(mask)) {
          idx = index;
          cache_hit(idx, mf->is_critical());
          return HIT;
        } else {
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
  }

  // Make prediction.
  unsigned start_way, end_way;
  if (CCBP[signature] >= 2) {
    start_way = 0;
    end_way = m_config.m_assoc * CRITICAL_PERCENTAGE;
    this->is_correct = mf->is_critical();
  } else {
    start_way = m_config.m_assoc * CRITICAL_PERCENTAGE;
    end_way = m_config.m_assoc;
    this->is_correct = !mf->is_critical();
  }

  // Looking for allocatable cache block.
  for (unsigned way = start_way; way < end_way; ++way) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];

    if (!line->is_reserved_line()) {
      all_reserved = false;
      if (line->is_invalid_line()) {
        invalid_line = index;
      } else {
        // valid line : keep track of most appropriate replacement candidate
        if (m_config.m_replacement_policy == LRU) {
          if (line->get_last_access_time() < valid_timestamp) {
            valid_timestamp = line->get_last_access_time();
            valid_line = index;
          }
        } else if (m_config.m_replacement_policy == FIFO) {
          if (line->get_alloc_time() < valid_timestamp) {
            valid_timestamp = line->get_alloc_time();
            valid_line = index;
          }
        }
      }
    }
  }

  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  if (invalid_line != (unsigned)-1) {
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line;
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  if (probe_mode && m_config.is_streaming()) {
    line_table::const_iterator i =
        pending_lines.find(m_config.block_addr(addr));
    assert(mf);
    if (!mf->is_write() && i != pending_lines.end()) {
      if (i->second != mf->get_inst().get_uid()) return SECTOR_MISS;
    }
  }

  return MISS;
}

enum cache_request_status cacp_tag_array::access(
    new_addr_type addr, unsigned time, unsigned &idx, bool &wb,
    evicted_block_info &evicted, mem_fetch *mf) {
  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  enum cache_request_status status =
      probe(addr, idx, mf->get_access_sector_mask(), false, mf);
  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
      break;
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size());
          // Update CACP predictor.
          evict_line(idx, /*set_index=*/ m_config.tag(addr));
        }
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void cacp_tag_array::cache_hit(bool critical, unsigned idx)
{
  // Ignore setting SRRIP promotion position here.
  unsigned signature = m_lines[idx]->m_signature;
  if (critical) {
    // Correct prediction, increment CCBP.
    m_lines[idx]->m_c_reuse = true;
    if (CCBP[signature] < 3) {
      CCBP[signature]++;
    }
    if (SHiP[signature] < 3) {
      SHiP[signature]++;
    }
  } else {
    // Hit is from non-critical warp.
  	m_lines[idx]->m_nc_reuse = true;
  	if (SHiP[signature] < 3) {
  	  SHiP[signature]++;
    }
  }
}

void cacp_tag_array::evict_line(unsigned idx, unsigned set_index)
{
  cache_block_t *evicted = m_lines[idx];
  unsigned signature = evicted->m_signature;
  unsigned way = idx - set_index * m_config.m_assoc;
  if (!evicted->m_c_reuse /*c_reuse == false*/ &&
      evicted->m_nc_reuse /*nc_reuse == true*/ &&
      way < m_config.m_assoc * CRITICAL_PERCENTAGE /*critial partition*/ &&
      CCBP[evicted->m_signature] > 0) {
    CCBP[evicted->m_signature]--;
  } else if (!evicted->m_c_reuse /*c_reuse == false*/ &&
             !evicted->m_nc_reuse /*nc_reuse == false*/ &&
             SHiP[evicted->m_signature] > 0) {
    SHiP[evicted->m_signature]--;
  }
}
