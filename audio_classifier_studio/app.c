/***************************************************************************//**
 * @file
 * @brief Top level application functions
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include "app.h"
#include "audio_classifier.h"
#include "sl_bt_api.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define APP_BLE_SIGNAL_NOTIFY_DETECTION (1u << 0)
#define APP_BLE_DEVICE_NAME "BabyDog-XG26"

// Custom 128-bit UUIDs in little-endian format.
static const uint8_t app_service_uuid[16] = {
  0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
  0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x10
};
static const uuid_128 app_char_uuid = {
  .data = {
    0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x60, 0x71, 0x82,
    0x93, 0xa4, 0xb5, 0xc6, 0xd7, 0xe8, 0xf9, 0x11
  }
};

static uint8_t advertising_set_handle = 0xff;
static uint8_t connection_handle = SL_BT_INVALID_CONNECTION_HANDLE;
static uint16_t detection_characteristic_handle = 0;
static bool notifications_enabled = false;

static volatile uint8_t pending_class_id = 2;
static volatile uint8_t pending_score = 0;
static volatile bool pending_detection = false;

static void app_check_status(sl_status_t sc, const char *msg)
{
  if (sc != SL_STATUS_OK) {
    printf("BLE error (%s): 0x%04lx\r\n", msg, (unsigned long)sc);
  }
}

static void app_ble_init_gatt(void)
{
  sl_status_t sc;
  uint16_t session = 0;
  uint16_t service = 0;
  uint8_t initial_value[2] = { 2, 0 };

  sc = sl_bt_gattdb_new_session(&session);
  app_check_status(sc, "gattdb_new_session");
  if (sc != SL_STATUS_OK) { return; }

  sc = sl_bt_gattdb_add_service(session,
                                sl_bt_gattdb_primary_service,
                                SL_BT_GATTDB_ADVERTISED_SERVICE,
                                sizeof(app_service_uuid),
                                app_service_uuid,
                                &service);
  app_check_status(sc, "gattdb_add_service");
  if (sc != SL_STATUS_OK) { return; }

  sc = sl_bt_gattdb_add_uuid128_characteristic(session,
                                               service,
                                               SL_BT_GATTDB_CHARACTERISTIC_READ
                                               | SL_BT_GATTDB_CHARACTERISTIC_NOTIFY,
                                               0,
                                               0,
                                               app_char_uuid,
                                               sl_bt_gattdb_fixed_length_value,
                                               sizeof(initial_value),
                                               sizeof(initial_value),
                                               initial_value,
                                               &detection_characteristic_handle);
  app_check_status(sc, "gattdb_add_char");
  if (sc != SL_STATUS_OK) { return; }

  sc = sl_bt_gattdb_start_service(session, service);
  app_check_status(sc, "gattdb_start_service");
  if (sc != SL_STATUS_OK) { return; }

  sc = sl_bt_gattdb_commit(session);
  app_check_status(sc, "gattdb_commit");
}

static void app_ble_start_advertising(void)
{
  sl_status_t sc;
  if (advertising_set_handle == 0xff) {
    sc = sl_bt_advertiser_create_set(&advertising_set_handle);
    app_check_status(sc, "advertiser_create_set");
    if (sc != SL_STATUS_OK) { return; }
  }

  sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                             sl_bt_advertiser_general_discoverable);
  app_check_status(sc, "advertiser_generate_data");
  if (sc != SL_STATUS_OK) { return; }

  // Put a stable human-readable name in scan response so the board is easy to find.
  {
    const char *name = APP_BLE_DEVICE_NAME;
    size_t name_len = strlen(name);
    if (name_len > 29) {
      name_len = 29;
    }
    uint8_t scan_rsp_data[31];
    scan_rsp_data[0] = (uint8_t)(name_len + 1); // AD structure length (type + data)
    scan_rsp_data[1] = 0x09;                    // Complete Local Name
    memcpy(&scan_rsp_data[2], name, name_len);

    sc = sl_bt_legacy_advertiser_set_data(advertising_set_handle,
                                          sl_bt_advertiser_scan_response_packet,
                                          (uint8_t)(name_len + 2),
                                          scan_rsp_data);
    app_check_status(sc, "advertiser_set_scan_rsp_name");
    if (sc != SL_STATUS_OK) { return; }
  }

  sc = sl_bt_advertiser_set_timing(advertising_set_handle,
                                   160, // 100 ms
                                   160,
                                   0,
                                   0);
  app_check_status(sc, "advertiser_set_timing");
  if (sc != SL_STATUS_OK) { return; }

  sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                     sl_bt_legacy_advertiser_connectable);
  app_check_status(sc, "legacy_advertiser_start");
}

/***************************************************************************//**
 * Initialize application.
 ******************************************************************************/
void app_init(void)
{
  audio_classifier_init();
}

void app_process_action(void)
{
}

void app_ble_report_detection(uint8_t class_id, uint8_t score)
{
  pending_class_id = class_id;
  pending_score = score;
  pending_detection = true;
  (void)sl_bt_external_signal(APP_BLE_SIGNAL_NOTIFY_DETECTION);
}

void sl_bt_on_event(sl_bt_msg_t *evt)
{
  sl_status_t sc;

  switch (SL_BT_MSG_ID(evt->header)) {
    case sl_bt_evt_system_boot_id:
      {
        bd_addr identity;
        uint8_t identity_type = 0;
        sc = sl_bt_system_get_identity_address(&identity, &identity_type);
        app_check_status(sc, "get_identity_address");
        if (sc == SL_STATUS_OK) {
          printf("BLE address: %02X:%02X:%02X:%02X:%02X:%02X (type=%u)\r\n",
                 identity.addr[5], identity.addr[4], identity.addr[3],
                 identity.addr[2], identity.addr[1], identity.addr[0],
                 identity_type);
        }
      }
      app_ble_init_gatt();
      app_ble_start_advertising();
      break;

    case sl_bt_evt_connection_opened_id:
      connection_handle = evt->data.evt_connection_opened.connection;
      notifications_enabled = false;
      break;

    case sl_bt_evt_connection_closed_id:
      connection_handle = SL_BT_INVALID_CONNECTION_HANDLE;
      notifications_enabled = false;
      app_ble_start_advertising();
      break;

    case sl_bt_evt_gatt_server_characteristic_status_id:
      if (evt->data.evt_gatt_server_characteristic_status.characteristic == detection_characteristic_handle
          && ((evt->data.evt_gatt_server_characteristic_status.status_flags & sl_bt_gatt_server_client_config) != 0)) {
        const uint16_t flags = evt->data.evt_gatt_server_characteristic_status.client_config_flags;
        notifications_enabled = ((flags & sl_bt_gatt_server_notification) != 0);
      }
      break;

    case sl_bt_evt_system_external_signal_id:
      if ((evt->data.evt_system_external_signal.extsignals & APP_BLE_SIGNAL_NOTIFY_DETECTION) != 0
          && pending_detection
          && connection_handle != SL_BT_INVALID_CONNECTION_HANDLE
          && notifications_enabled) {
        uint8_t payload[2];
        payload[0] = pending_class_id;
        payload[1] = pending_score;
        pending_detection = false;

        sc = sl_bt_gatt_server_write_attribute_value(detection_characteristic_handle,
                                                     0,
                                                     sizeof(payload),
                                                     payload);
        app_check_status(sc, "gatt_write_attr");
        if (sc != SL_STATUS_OK) { break; }

        sc = sl_bt_gatt_server_send_notification(connection_handle,
                                                 detection_characteristic_handle,
                                                 sizeof(payload),
                                                 payload);
        app_check_status(sc, "gatt_notify");
      }
      break;

    default:
      break;
  }
}
