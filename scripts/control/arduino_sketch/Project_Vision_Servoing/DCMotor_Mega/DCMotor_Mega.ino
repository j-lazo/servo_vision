//Combining two motor controllers into one Mega board

//This is the section for encoder and ISR
int encoderPin_upper_1 = 18; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin_upper_2 = 19; //Encoder Otput 'B' must connected with intreput pin of arduino.

int encoderPin_side_1 = 20; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin_side_2 = 21; //Encoder Otput 'B' must connected with intreput pin of arduino.

volatile int lastEncoded_upper = 0; // Here updated value of encoder store.
volatile int encoderValue_upper = 0; // Raw encoder value
volatile int lastEncoded_side = 0; // Here updated value of encoder store.
volatile int encoderValue_side = 0; // Raw encoder value

int PPR = 2;  // Encoder Pulse per revolution.
int lastMSB_upper = 0;
int lastLSB_upper = 0;
int lastMSB_side = 0;
int lastLSB_side = 0;

// Initialize PID controller
#include <PID_v1.h>
// PID Upper motor contrller
#define UpperMotEnable 5 //Motor Enamble pin Runs on PWM signal
#define UpperMotFwd  6  // Motor Forward pin
#define UpperMotRev  7 // Motor Reverse pin

double upper_kp = 2.2 , upper_ki = 2.8 , upper_kd = 0.013;
double upper_input = 0, upper_output = 0, upper_setpoint = 0;
double REV_upper = 0;
int user_input_upper = 0;
PID upper_motor_PID(&upper_input, &upper_output, &upper_setpoint, upper_kp, upper_ki, upper_kd, DIRECT);

// PID Side motor contrller
#define SideMotEnable 8 //Motor Enamble pin Runs on PWM signal
#define SideMotFwd  9  // Motor Forward pin
#define SideMotRev  10 // Motor Reverse pin

double side_kp = 2.2 , side_ki = 2.8 , side_kd = 0.013;
double side_input = 0, side_output = 0, side_setpoint = 0;
double REV_side = 0;
int user_input_side = 0;
PID side_motor_PID(&side_input, &side_output, &side_setpoint, side_kp, side_ki, side_kd, DIRECT);

// analog read pin:
int analogPin1 = A1;
int analogPin2 = A2;

// Serial initialization

char receiveData = 'c';
String receiveData_upper = "";
String receiveData_side = "";

void setup() {
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);

  //Seup for Serial connection:
  Serial.begin(115200); // set the baud rate
  Serial.flush();
  
  
  //Setup: interrupts for motors and encoders
  pinMode(encoderPin_upper_1, INPUT_PULLUP); 
  pinMode(encoderPin_upper_2, INPUT_PULLUP);
  pinMode(encoderPin_side_1, INPUT_PULLUP); 
  pinMode(encoderPin_side_2, INPUT_PULLUP);
  digitalWrite(encoderPin_upper_1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_upper_2, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_side_1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_side_2, HIGH); //turn pullup resistor on
  attachInterrupt(digitalPinToInterrupt(encoderPin_upper_1), updateEncoder_upper, CHANGE); 
  attachInterrupt(digitalPinToInterrupt(encoderPin_upper_2), updateEncoder_upper, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin_side_1), updateEncoder_side, CHANGE); 
  attachInterrupt(digitalPinToInterrupt(encoderPin_side_2), updateEncoder_side, CHANGE);

  //Setup for Motor
  //MotorSetup:
  pinMode(UpperMotEnable, OUTPUT);
  pinMode(UpperMotFwd, OUTPUT); 
  pinMode(UpperMotRev, OUTPUT); 
  TCCR1B = TCCR1B & 0b11111000 | 1;  // set 31KHz PWM to prevent motor noise
  upper_motor_PID.SetMode(AUTOMATIC);   //set PID in Auto mode
  upper_motor_PID.SetSampleTime(1);  // refresh rate of PID controller
  upper_motor_PID.SetOutputLimits(-150, 150); // this is the MAX PWM value to move motor, here change in value reflect change in speed of motor.
  pinMode(SideMotEnable, OUTPUT);
  pinMode(SideMotFwd, OUTPUT); 
  pinMode(SideMotRev, OUTPUT); 
  TCCR1B = TCCR1B & 0b11111000 | 1;  // set 31KHz PWM to prevent motor noise
  side_motor_PID.SetMode(AUTOMATIC);   //set PID in Auto mode
  side_motor_PID.SetSampleTime(1);  // refresh rate of PID controller
  side_motor_PID.SetOutputLimits(-150, 150); // this is the MAX PWM value to move motor, here change in value reflect change in speed of motor.

  delay(1000);

}

void loop() {
//  put your main code here, to run repeatedly:
//  Serial.print(encoderValue_upper);
//  Serial.print(",");
//  Serial.println(encoderValue_side);

  //Read from Serial
  int seperate_flag = 0;
  while(Serial.available()){ // only send data back if data has been sent
    delay(3);
    char c = Serial.read(); // read the incoming data
//    Serial.print(c);
    if(c == ','){
      seperate_flag = 1;
    }
    else if(c == 'r'){
      receiveData = 'r';
    }
    else{
      if(seperate_flag == 0) receiveData_upper += c;
      else if(seperate_flag == 1) receiveData_side += c;
    }
  }
  if (receiveData == 'r'){
//    Serial.print("(");
    Serial.println(encoderValue_upper);
//    Serial.print(",");
    Serial.println(encoderValue_side);
//    Serial.println(")");
    receiveData = 'c'; // clear the flag comes into here
  }
  else if (receiveData_upper.length() != 0 && receiveData_side.length() != 0){
    user_input_upper = receiveData_upper.toInt();
    user_input_side  = receiveData_side.toInt();
    receiveData_upper = "";
    receiveData_side = "";
//    Serial.println("WHY?");
  }
  else{
    receiveData_upper = "";
    receiveData_side = "";
  }
//  Serial.print("(");
//  Serial.print(user_input_upper);
//  Serial.print(",");
//  Serial.print(user_input_side);
//  Serial.println(")");
//  receiveData_upper = "";
//  receiveData_side = "";
  delay(5);
  
  //Read input data
//  user_input_upper = analogRead(analogPin1);
//  user_input_side = analogRead(analogPin2);
  receiveData_upper = "";
  receiveData_side = "";
  REV_upper = map (user_input_upper, -90, 90, -2400, 2400 ); // mapping degree into pulse
  REV_side = map (user_input_side, -90, 90, -2400, 2400 ); // mapping degree into pulse
  

  //PID controller + PWM output
  //upper motor
  upper_setpoint = REV_upper;                  // set a new setpoint from python server
  upper_input = encoderValue_upper;
  upper_motor_PID.Compute();                 // calculate new output
  pwmOut(upper_output, UpperMotEnable);
  //side motor
  side_setpoint = REV_side;                  // set a new setpoint from python server
  side_input = encoderValue_side;
  side_motor_PID.Compute();                 // calculate new output
  pwmOut(side_output, SideMotEnable);
//  for serial plot test
//  Serial.print(REV_upper);
//  Serial.print(",");
//  Serial.print(encoderValue_upper);
//  Serial.print(",");
//  Serial.print(REV_side);
//  Serial.print(",");
//  Serial.println(encoderValue_side);
}

void updateEncoder_upper(){
  int MSB_upper = digitalRead(encoderPin_upper_1); //MSB = most significant bit
  int LSB_upper = digitalRead(encoderPin_upper_2); //LSB = least significant bit

  int encoded_upper = (MSB_upper << 1) |LSB_upper; //converting the 2 pin value to single number
  int sum_upper  = (lastEncoded_upper << 2) | encoded_upper; //adding it to the previous encoded value

  if(sum_upper == 0b1101 || sum_upper == 0b0100 || sum_upper == 0b0010 || sum_upper == 0b1011) encoderValue_upper ++;
  if(sum_upper == 0b1110 || sum_upper == 0b0111 || sum_upper == 0b0001 || sum_upper == 0b1000) encoderValue_upper --;

  lastEncoded_upper = encoded_upper; //store this value for next time

}

void updateEncoder_side(){
  int MSB_side = digitalRead(encoderPin_side_1); //MSB = most significant bit
  int LSB_side = digitalRead(encoderPin_side_2); //LSB = least significant bit

  int encoded_side = (MSB_side << 1) |LSB_side; //converting the 2 pin value to single number
  int sum_side  = (lastEncoded_side << 2) | encoded_side; //adding it to the previous encoded value

  if(sum_side == 0b1101 || sum_side == 0b0100 || sum_side == 0b0010 || sum_side == 0b1011) encoderValue_side ++;
  if(sum_side == 0b1110 || sum_side == 0b0111 || sum_side == 0b0001 || sum_side == 0b1000) encoderValue_side --;

  lastEncoded_side = encoded_side; //store this value for next time

}

void pwmOut(int out, int motor_number) {   // motor number 1 == upper // 2 == side
  if (motor_number == 5){                        
    if (out > 0) {                         // if REV > encoderValue motor move in forward direction.    
      analogWrite(UpperMotEnable, out);         // Enabling motor enable pin to reach the desire angle
      forward(motor_number);                           // calling motor to move forward
    }
    else {
      analogWrite(UpperMotEnable, abs(out));          // if REV < encoderValue motor move in forward direction.                      
      reverse(motor_number);                            // calling motor to move reverse
    }
//  receiveData=""; // Cleaning User input, ready for new Input
  }
  else if(motor_number == 8){
    if (out > 0) {                         // if REV > encoderValue motor move in forward direction.    
      analogWrite(SideMotEnable, out);         // Enabling motor enable pin to reach the desire angle
      forward(motor_number);                           // calling motor to move forward
    }
    else {
      analogWrite(SideMotEnable, abs(out));          // if REV < encoderValue motor move in forward direction.                      
      reverse(motor_number);                            // calling motor to move reverse
    }
  }
}

void forward (int motor_number) {
  if(motor_number == 5){
    digitalWrite(UpperMotFwd, HIGH); 
    digitalWrite(UpperMotRev, LOW); 
  }
  else if(motor_number == 8){
    digitalWrite(SideMotFwd, HIGH); 
    digitalWrite(SideMotRev, LOW);     
  }

}

void reverse (int motor_number) {
  if(motor_number == 5){
    digitalWrite(UpperMotFwd, LOW); 
    digitalWrite(UpperMotRev, HIGH); 
  }
  else if(motor_number == 8){
    digitalWrite(SideMotFwd, LOW); 
    digitalWrite(SideMotRev, HIGH);     
  }
}
void finish (int motor_number) {
  if(motor_number == 5){
    digitalWrite(UpperMotFwd, LOW); 
    digitalWrite(UpperMotRev, LOW); 
  }
  else if(motor_number == 8){
    digitalWrite(SideMotFwd, LOW); 
    digitalWrite(SideMotRev, LOW);     
  }
  
}
