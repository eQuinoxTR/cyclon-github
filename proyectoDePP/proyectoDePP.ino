#include <ESP32Servo.h>;
Servo plato;
String label = "";

// menos ms: mas rapido
int velocidad = 7; 
int tiempoDePausa = 1000;

void setup() {
  Serial.begin(9600);
  plato.attach(18);
  plato.write(90);
}

void loop() {
  if (Serial.available() > 0) {
    label = Serial.readStringUntil('\n');
    moverPlato(label);
    delay(1000);
    label = "";
    limpiarBuffer();
  }
}

void moverPlato(String label) {
  if (label == "botella" || label == "lata_aluminio" || label == "carton_limpio" || label == "vidrio" || label == "papel_limpio") {
    animacionReciclable();
  } else if (label == "papel_sucio" || label == "plastico_sucio" || label == "comida") {
    animacionNoReciclable();
  }
}

void limpiarBuffer() {
    while (Serial.available() > 0) {
      Serial.read(); // Lee un byte y no lo guarda en ningún lado
    }
}


// [ Animaciones ] 

void animacionReciclable() {
  int grados;
  for (grados = 90; grados <= 180; grados++) {
    delay(velocidad);
    plato.write(grados);
  }

  delay(tiempoDePausa);

  for (grados = 180; grados >= 90; grados--) {
    delay(velocidad);
    plato.write(grados);
  }
}

void animacionNoReciclable() {
  int grados;
  for (grados = 90; grados >= 0; grados--) {
    delay(velocidad);
    plato.write(grados);
  }

  delay(tiempoDePausa);

  for (grados = 0; grados <= 90; grados++) {
    delay(velocidad);
    plato.write(grados);
  } 
}