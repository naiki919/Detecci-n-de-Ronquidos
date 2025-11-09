/*
  Snore Streamer – Arduino Leonardo (ATmega32u4) @ 16 kHz
  - Muestreo ADC estable a 16000 Hz con Timer1
  - Protocolo binario compatible con la Raspberry
  - Comando 'B' + 2 bytes (duración ms, LE) -> beep
*/

#include <Arduino.h>

// ====== CONFIGURACIÓN ======
const uint8_t  MIC_PIN_ADC_CH = 7;     // A0 en Leonardo = ADC7 (PF7)
const uint8_t  BUZZER_PIN     = 9;     // pin del buzzer
const uint32_t SAMPLE_RATE    = 16000; // Hz
const uint16_t PACKET_SAMPLES = 256;   // 16 ms de audio por paquete
const int16_t  CENTER         = 512;   // nivel medio ADC 10-bit
const int16_t  SCALE          = 64;    // ganancia (ajusta volumen entrada)
// ============================

// Buffer circular para transmisión
volatile int16_t sampleBuffer[PACKET_SAMPLES];
volatile uint16_t wrIdx = 0;
volatile bool packetReady = false;

// ====== ADC + Timer1 @16kHz ======
void setupADC() {
  // Canal ADC7 (A0) con referencia AVcc
  ADMUX  = (1 << REFS0) | (MIC_PIN_ADC_CH & 0x07);
  ADCSRB = 0;
  // Prescaler 64 → F_ADC=250 kHz → conversión ≈52 µs (<62.5 µs de 16kHz)
  ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1);
  ADCSRA |= (1 << ADSC); // primera conversión
}

void setupTimer1() {
  cli();
  TCCR1A = 0;
  TCCR1B = 0;
  // CTC con prescaler=1 → OCR1A=999 → 16000 Hz
  TCCR1B = (1 << WGM12) | (1 << CS10);
  OCR1A = 999;
  TIMSK1 = (1 << OCIE1A);
  sei();
}

// ISR a 16 kHz
ISR(TIMER1_COMPA_vect) {
  if (!(ADCSRA & (1 << ADSC))) {
    uint8_t low  = ADCL;
    uint8_t high = ADCH;
    uint16_t raw = (high << 8) | low;
    int16_t s = (int16_t)((int32_t)(raw - CENTER) * SCALE);

    sampleBuffer[wrIdx++] = s;
    if (wrIdx >= PACKET_SAMPLES) {
      wrIdx = 0;
      packetReady = true;
    }
  }
  ADCSRA |= (1 << ADSC);
}

// ====== ENVÍO DE PAQUETES ======
void sendPacket(int16_t *data, uint16_t n) {
  uint8_t hdr[4] = {0xAA, 0x55, (uint8_t)(n & 0xFF), (uint8_t)((n >> 8) & 0xFF)};
  Serial.write(hdr, 4);
  Serial.write((uint8_t*)data, n * sizeof(int16_t));
}

// ====== SETUP ======
void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 1000)) {}

  setupADC();
  setupTimer1();

  tone(BUZZER_PIN, 2500, 120); // beep de arranque
  delay(150);
}

// ====== LOOP PRINCIPAL ======
void loop() {
  // Leer comando desde la Raspberry
  while (Serial.available() >= 1) {
    int c = Serial.read();
    if (c == 'B') {
      // Leer duración (2 bytes LE)
      uint16_t ms = 600;
      unsigned long t0 = millis();
      while (Serial.available() < 2 && (millis() - t0) < 10) {}
      if (Serial.available() >= 2) {
        uint8_t lo = Serial.read();
        uint8_t hi = Serial.read();
        ms = (uint16_t)(lo | (hi << 8));
      }

      // Si el buzzer es ACTIVO (ya oscila al aplicar 5V)
      digitalWrite(BUZZER_PIN, HIGH);
      delay(ms);
      digitalWrite(BUZZER_PIN, LOW);

      // Si el buzzer es PASIVO, comenta las 3 líneas anteriores y descomenta:
      // tone(BUZZER_PIN, 4000, ms);
    }
  }

  // Enviar audio cuando el buffer esté lleno
  if (packetReady) {
    noInterrupts();
    static int16_t localBuf[PACKET_SAMPLES];
    for (uint16_t i = 0; i < PACKET_SAMPLES; ++i)
      localBuf[i] = sampleBuffer[i];
    packetReady = false;
    interrupts();

    sendPacket(localBuf, PACKET_SAMPLES);
  }
}
