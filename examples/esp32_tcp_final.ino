// 这个文件是对方跑的文件

// ==================== 每块板只改这里 ====================
#define NODE_INDEX  0     // 0=骨盆, 1=左手腕, 2=右手腕, 3=左脚踝, 4=右脚踝, 5=头部
#define DEVICE_ID   "node01" // 对应 node01~node06
// =======================================================

// ==================== 网络配置 ====================
const char* ssid       = "Jiaojian";
const char* password   = "12345678";
const char* SERVER_IP  = "49.234.57.210";  // 你的公网IP
const int   SERVER_PORT = 8001;             // frp 映射端口 (本地9001 → 公网8001)
// ==================================================

// ==================== IMU I2C 配置 ====================
#define IMU_ADDR 0x23
#define SDA_PIN  8
#define SCL_PIN  9
// ======================================================

#include <WiFi.h>
#include <Wire.h>

WiFiClient tcpClient;

// ---------- I2C 辅助 ----------
bool imuReadBytes(uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)IMU_ADDR, (int)len) != len) return false;
  for (uint8_t i = 0; i < len; i++) {
    if (!Wire.available()) return false;
    buf[i] = Wire.read();
  }
  return true;
}

int16_t toInt16LE(uint8_t lo, uint8_t hi) { return (int16_t)((hi << 8) | lo); }

bool imuReadAccelerometer(float out[3]) {
  uint8_t raw[6];
  if (!imuReadBytes(0x04, raw, 6)) return false;
  float scale = 16.0f / 32767.0f * 9.81f;  // → m/s²
  out[0] = toInt16LE(raw[0], raw[1]) * scale;
  out[1] = toInt16LE(raw[2], raw[3]) * scale;
  out[2] = toInt16LE(raw[4], raw[5]) * scale;
  return true;
}

bool imuReadGyroscope(float out[3]) {
  uint8_t raw[6];
  if (!imuReadBytes(0x0A, raw, 6)) return false;
  float scale = 2000.0f / 32767.0f;  // → deg/s
  out[0] = toInt16LE(raw[0], raw[1]) * scale;
  out[1] = toInt16LE(raw[2], raw[3]) * scale;
  out[2] = toInt16LE(raw[4], raw[5]) * scale;
  return true;
}

// ---------- 互补滤波器（板上计算欧拉角） ----------
float roll = 0, pitch = 0, yaw = 0;
unsigned long lastUs = 0;
const float ALPHA = 0.98f;

void updateAttitude(float acc[3], float gyro_degs[3]) {
  unsigned long now = micros();
  float dt = (now - lastUs) / 1e6f;
  lastUs = now;
  if (dt <= 0 || dt > 1.0f) return;

  float ax = acc[0], ay = acc[1], az = acc[2];
  float norm = sqrtf(ax*ax + ay*ay + az*az);
  if (norm < 0.5f) return;
  ax /= norm; ay /= norm; az /= norm;

  float roll_acc  = atan2f(ay, az) * 180.0f / M_PI;
  float pitch_acc = atan2f(-ax, sqrtf(ay*ay + az*az)) * 180.0f / M_PI;

  roll  = ALPHA * (roll  + gyro_degs[0] * dt) + (1.0f - ALPHA) * roll_acc;
  pitch = ALPHA * (pitch + gyro_degs[1] * dt) + (1.0f - ALPHA) * pitch_acc;
  yaw   = yaw + gyro_degs[2] * dt;
}

// ---------- WiFi ----------
void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println();
  Serial.print("WiFi OK  IP: "); Serial.println(WiFi.localIP());
}

// ---------- TCP ----------
void ensure_connected() {
  if (tcpClient.connected()) return;
  Serial.printf("TCP connecting %s:%d ...\n", SERVER_IP, SERVER_PORT);
  while (!tcpClient.connect(SERVER_IP, SERVER_PORT)) {
    Serial.println("  failed, retry in 2s");
    delay(2000);
  }
  Serial.println("TCP connected!");
}

// ---------- 发送帧 ----------
// 协议: {"node":0,"t":12.3,"acc":[ax,ay,az],"rpy":[roll,pitch,yaw]}\n
void sendFrame(float acc[3]) {
  char buf[192];
  int len = snprintf(buf, sizeof(buf),
    "{\"node\":%d,\"t\":%.3f,\"acc\":[%.4f,%.4f,%.4f],\"rpy\":[%.3f,%.3f,%.3f]}\n",
    NODE_INDEX,
    millis() / 1000.0f,
    acc[0], acc[1], acc[2],
    roll, pitch, yaw
  );
  tcpClient.write((uint8_t*)buf, len);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.printf("\n=== ESP32 TCP  node=%d ===\n", NODE_INDEX);

  setup_wifi();
  Wire.begin(SDA_PIN, SCL_PIN, 100000);
  delay(500);

  lastUs = micros();
  ensure_connected();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) setup_wifi();
  ensure_connected();

  float acc[3], gyro[3];
  if (imuReadAccelerometer(acc) && imuReadGyroscope(gyro)) {
    updateAttitude(acc, gyro);
    sendFrame(acc);
    Serial.printf("[node%d] rpy=%.1f,%.1f,%.1f\n", NODE_INDEX, roll, pitch, yaw);
  } else {
    Serial.println("IMU failed");
  }

  delay(50);  // 20 Hz
}
