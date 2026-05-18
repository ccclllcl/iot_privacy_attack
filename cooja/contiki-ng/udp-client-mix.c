#include "udp-metric-common.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "random.h"

#ifndef DUMMY_ENABLED
#define DUMMY_ENABLED 1
#endif

#ifndef DUMMY_PROB_PCT
#define DUMMY_PROB_PCT 30
#endif

#ifndef JITTER_SEC
#define JITTER_SEC 2
#endif

#ifndef ADAPTIVE_MODE
#define ADAPTIVE_MODE 0
#endif

static struct simple_udp_connection udp_conn;
static uint32_t rx_count;

PROCESS(udp_client_process, "UDP metric mixed-traffic client");
AUTOSTART_PROCESSES(&udp_client_process);

static void
udp_rx_callback(struct simple_udp_connection *c,
                const uip_ipaddr_t *sender_addr,
                uint16_t sender_port,
                const uip_ipaddr_t *receiver_addr,
                uint16_t receiver_port,
                const uint8_t *data,
                uint16_t datalen)
{
  (void)c;
  (void)sender_port;
  (void)receiver_addr;
  (void)receiver_port;
  LOG_INFO("Received response bytes=%u from ", datalen);
  LOG_INFO_6ADDR(sender_addr);
  LOG_INFO_("\n");
  rx_count++;
}

static uint8_t
current_dummy_prob(void)
{
#if ADAPTIVE_MODE
  clock_time_t now = clock_time();
  if(now < (5 * 60 * CLOCK_SECOND)) {
    return 15;
  }
  if(now < (10 * 60 * CLOCK_SECOND)) {
    return 45;
  }
  return 25;
#else
  return DUMMY_PROB_PCT;
#endif
}

PROCESS_THREAD(udp_client_process, ev, data)
{
  static struct etimer periodic_timer;
  static metric_packet_t pkt;
  static metric_packet_t dummy_pkt;
  static uint32_t tx_count;
  static uint32_t missed_tx_count;
  static uint32_t dummy_count;
  uip_ipaddr_t dest_ipaddr;

  PROCESS_BEGIN();

  simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL, UDP_SERVER_PORT, udp_rx_callback);

  etimer_set(&periodic_timer, random_rand() % SEND_INTERVAL);
  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

    if(NETSTACK_ROUTING.node_is_reachable() &&
       NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr)) {
      if(tx_count % 10 == 0) {
        LOG_INFO("Tx/Rx/MissedTx: %" PRIu32 "/%" PRIu32 "/%" PRIu32 "\n",
                 tx_count, rx_count, missed_tx_count);
        metric_log_energest();
      }

      metric_fill_packet(&pkt, METRIC_PACKET_REAL, tx_count, "hello");
      simple_udp_sendto(&udp_conn, &pkt, sizeof(pkt), &dest_ipaddr);
      LOG_INFO("METRIC_TX type=REAL node=%u seq=%" PRIu32 " bytes=%u time_ms=%" PRIu32 "\n",
               node_id, tx_count, (unsigned)sizeof(pkt), pkt.send_time_ms);
      tx_count++;

#if DUMMY_ENABLED
      if((random_rand() % 100) < current_dummy_prob()) {
        metric_fill_packet(&dummy_pkt, METRIC_PACKET_DUMMY, dummy_count, "status");
        simple_udp_sendto(&udp_conn, &dummy_pkt, sizeof(dummy_pkt), &dest_ipaddr);
        LOG_INFO("METRIC_TX type=DUMMY node=%u seq=%" PRIu32 " bytes=%u time_ms=%" PRIu32 "\n",
                 node_id, dummy_count, (unsigned)sizeof(dummy_pkt), dummy_pkt.send_time_ms);
        dummy_count++;
      }
#endif
    } else {
      LOG_INFO("Not reachable yet\n");
      if(tx_count > 0) {
        missed_tx_count++;
      }
    }

    {
      int jitter = (int)(random_rand() % (2 * JITTER_SEC + 1)) - JITTER_SEC;
      clock_time_t next = SEND_INTERVAL + (clock_time_t)(jitter * CLOCK_SECOND);
      if(next < CLOCK_SECOND) {
        next = CLOCK_SECOND;
      }
      etimer_set(&periodic_timer, next);
    }
  }

  PROCESS_END();
}
