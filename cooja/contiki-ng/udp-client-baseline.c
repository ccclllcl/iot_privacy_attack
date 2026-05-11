#include "udp-metric-common.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "random.h"

static struct simple_udp_connection udp_conn;
static uint32_t rx_count;

PROCESS(udp_client_process, "UDP metric baseline client");
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

PROCESS_THREAD(udp_client_process, ev, data)
{
  static struct etimer periodic_timer;
  static metric_packet_t pkt;
  static uint32_t tx_count;
  static uint32_t missed_tx_count;
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
    } else {
      LOG_INFO("Not reachable yet\n");
      if(tx_count > 0) {
        missed_tx_count++;
      }
    }

    etimer_set(&periodic_timer, SEND_INTERVAL - CLOCK_SECOND + (random_rand() % (2 * CLOCK_SECOND)));
  }

  PROCESS_END();
}
