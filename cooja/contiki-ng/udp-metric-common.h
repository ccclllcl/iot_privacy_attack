#ifndef UDP_METRIC_COMMON_H_
#define UDP_METRIC_COMMON_H_

#include "contiki.h"
#include "net/ipv6/simple-udp.h"
#include "sys/energest.h"
#include "sys/log.h"
#include "sys/node-id.h"

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#ifndef LOG_MODULE
#define LOG_MODULE "Metric"
#endif

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

#define WITH_SERVER_REPLY 1
#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678
#define SEND_INTERVAL (10 * CLOCK_SECOND)

#define METRIC_MAGIC 0xC0A1u
#define METRIC_VERSION 1u
#define METRIC_PACKET_REAL 1u
#define METRIC_PACKET_DUMMY 2u
#define METRIC_PAYLOAD_BYTES 32u

typedef struct __attribute__((packed)) metric_packet {
  uint16_t magic;
  uint8_t version;
  uint8_t packet_type;
  uint16_t node_id;
  uint32_t seq;
  uint32_t send_time_ms;
  uint16_t payload_len;
  char payload[METRIC_PAYLOAD_BYTES];
} metric_packet_t;

static uint32_t
metric_now_ms(void)
{
  return (uint32_t)(((uint64_t)clock_time() * 1000ull) / CLOCK_SECOND);
}

static inline const char *
metric_type_name(uint8_t packet_type)
{
  if(packet_type == METRIC_PACKET_REAL) {
    return "REAL";
  }
  if(packet_type == METRIC_PACKET_DUMMY) {
    return "DUMMY";
  }
  return "UNKNOWN";
}

static inline void
metric_fill_packet(metric_packet_t *pkt, uint8_t packet_type, uint32_t seq, const char *label)
{
  size_t len;
  memset(pkt, 0, sizeof(*pkt));
  pkt->magic = METRIC_MAGIC;
  pkt->version = METRIC_VERSION;
  pkt->packet_type = packet_type;
  pkt->node_id = node_id;
  pkt->seq = seq;
  pkt->send_time_ms = metric_now_ms();
  snprintf(pkt->payload, sizeof(pkt->payload), "%s %" PRIu32, label, seq);
  len = strlen(pkt->payload);
  pkt->payload_len = (uint16_t)len;
}

static inline int
metric_packet_valid(const uint8_t *data, uint16_t datalen)
{
  const metric_packet_t *pkt;
  if(datalen < sizeof(metric_packet_t)) {
    return 0;
  }
  pkt = (const metric_packet_t *)data;
  return pkt->magic == METRIC_MAGIC && pkt->version == METRIC_VERSION;
}

static void
metric_log_energest(void)
{
  uint64_t cpu_ticks;
  uint64_t lpm_ticks;
  uint64_t tx_ticks;
  uint64_t rx_ticks;
  uint64_t total_ticks;

  energest_flush();
  cpu_ticks = energest_type_time(ENERGEST_TYPE_CPU);
  lpm_ticks = energest_type_time(ENERGEST_TYPE_LPM);
  tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
  rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
  total_ticks = cpu_ticks + lpm_ticks + tx_ticks + rx_ticks;

  LOG_INFO("ENERGEST node=%u cpu_ticks=%llu lpm_ticks=%llu tx_ticks=%llu rx_ticks=%llu total_ticks=%llu time_ms=%" PRIu32 "\n",
           node_id,
           (unsigned long long)cpu_ticks,
           (unsigned long long)lpm_ticks,
           (unsigned long long)tx_ticks,
           (unsigned long long)rx_ticks,
           (unsigned long long)total_ticks,
           metric_now_ms());
}

#endif
