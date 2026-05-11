#include "udp-metric-common.h"
#include "net/routing/routing.h"
#include "net/netstack.h"

static struct simple_udp_connection udp_conn;
static uint32_t rx_count;

PROCESS(udp_server_process, "UDP metric server");
AUTOSTART_PROCESSES(&udp_server_process);

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

  if(metric_packet_valid(data, datalen)) {
    const metric_packet_t *pkt = (const metric_packet_t *)data;
    uint32_t recv_ms = metric_now_ms();
    LOG_INFO("METRIC_RX type=%s src=%u seq=%" PRIu32 " send_ms=%" PRIu32 " recv_ms=%" PRIu32 " bytes=%u\n",
             metric_type_name(pkt->packet_type),
             pkt->node_id,
             pkt->seq,
             pkt->send_time_ms,
             recv_ms,
             datalen);
  } else {
    LOG_INFO("Received legacy request bytes=%u from ", datalen);
    LOG_INFO_6ADDR(sender_addr);
    LOG_INFO_("\n");
  }

  rx_count++;
  if(rx_count % 10 == 0) {
    metric_log_energest();
  }

#if WITH_SERVER_REPLY
  simple_udp_sendto(&udp_conn, data, datalen, sender_addr);
#endif
}

PROCESS_THREAD(udp_server_process, ev, data)
{
  PROCESS_BEGIN();

  NETSTACK_ROUTING.root_start();
  simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL, UDP_CLIENT_PORT, udp_rx_callback);

  PROCESS_END();
}
