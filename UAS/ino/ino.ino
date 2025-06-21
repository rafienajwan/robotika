char myData = 0;

// Motor Kiri
int motor1Pin1 = 22; 
int motor1Pin2 = 23; 
int enable1Pin = 21; 
// Motor Kanan
int motor2Pin1 = 17; 
int motor2Pin2 = 18; 
int enable2Pin = 16; 
// Setting PWM properties
const int freq = 30000;
const int pwmChannel = 0;
const int pwmChanne2 = 0;
const int resolution = 8;
const int dutyCycle = 255;

void setup() {
  Serial.begin(115200);
  // sets the pins as outputs:
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enable2Pin, OUTPUT);
  
  // Konfigurasikan fungsi PWM
  ledcSetup(pwmChannel, freq, resolution);
  ledcSetup(pwmChanne2, freq, resolution);

  // menentukan saluran ke GPIO untuk dikontrol
  ledcAttachPin(enable1Pin, pwmChannel);
  ledcAttachPin(enable2Pin, pwmChannel);
}
void loop()
{
myData = int(Serial.read());
if (myData == '1'){  
  Serial.println("Robot Maju Rek !!!");  
  // Maju
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  delay(200);
  ledcWrite(pwmChannel, dutyCycle);
}

else if (myData == '2'){  
  Serial.println("Robot Belok Kanan Rek!!!");  
  // Belok kanan
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  delay(200);
  ledcWrite(pwmChannel, dutyCycle);
}

else if (myData == '3'){  
  Serial.println("Robot belok Kiri Rek..!!!");  
  // Belok kiri
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);  
  delay(200);
  ledcWrite(pwmChannel, dutyCycle);
}
else if (myData == '4'){  
  Serial.println("Robot Mundur Rek !!!");  
  // Mundur
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);  
  delay(200);
  ledcWrite(pwmChannel, dutyCycle);
}


else if (myData == '0'){  
  Serial.println("Robot Berhenti Rek.. !!!");  
  // Belok kiri
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
  ledcWrite(pwmChannel, dutyCycle);
}
}