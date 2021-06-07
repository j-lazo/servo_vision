//Combining two motor controllers into one Mega board

//This is the section for encoder and ISR
int encoderPin_upper_1 = 18; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin_upper_2 = 19; //Encoder Otput 'B' must connected with intreput pin of arduino.

int encoderPin_side_1 = 20; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin_side_2 = 21; //Encoder Otput 'B' must connected with intreput pin of arduino.

int encoderPin_stepper_1 = 2; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin_stepper_2 = 3; //Encoder Otput 'B' must connected with intreput pin of arduino.

volatile int lastEncoded_upper = 0; // Here updated value of encoder store.
volatile int encoderValue_upper = 0; // Raw encoder value
volatile int lastEncoded_side = 0; // Here updated value of encoder store.
volatile int encoderValue_side = 0; // Raw encoder value
volatile int lastEncoded_stepper = 0; // Here updated value of encoder store.
volatile long encoderValue_stepper = 0; // Raw encoder value
int robot_boundary = 2500;



int PPR = 2;  // Encoder Pulse per revolution.
int lastMSB_upper = 0;
int lastLSB_upper = 0;
int lastMSB_side = 0;
int lastLSB_side = 0;
int lastMSB_stepper = 0;
int lastLSB_stepper = 0;

//////////////////////////// Initialize PID controller///////////////////////////////////////
#include <PID_v1.h>
// PID Upper motor controller
#define UpperMotEnable 5 //Motor Enamble pin Runs on PWM signal
#define UpperMotFwd  6  // Motor Forward pin
#define UpperMotRev  7 // Motor Reverse pin

unsigned long new_time = 0;
unsigned long old_time = 0;

long new_position_upper = 0;
long old_position_upper = 0;
long new_position_side = 0;
long old_position_side = 0;

long vel_upper;
long vel_side;
long timeArray[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int indexArray = 0;


double upper_kp = 2.4 , upper_ki = 40 , upper_kd = 0;
double upper_input = 0, upper_output = 0, upper_setpoint = 0;
long REV_upper = 0;
int user_input_upper = 0;
PID upper_motor_PID(&upper_input, &upper_output, &upper_setpoint, upper_kp, upper_ki, upper_kd, DIRECT);

// PID Side motor contrller
#define SideMotEnable 8 //Motor Enamble pin Runs on PWM signal
#define SideMotFwd  9  // Motor Forward pin
#define SideMotRev  10 // Motor Reverse pin

double side_kp = 1.85 , side_ki = 30 , side_kd = 0.0001;
double side_input = 0, side_output = 0, side_setpoint = 0;
long REV_side = 0;
int user_input_side = 0;
PID side_motor_PID(&side_input, &side_output, &side_setpoint, side_kp, side_ki, side_kd, DIRECT);

//only use for stepper controller
int user_input_stepper = 0;
double DIST_stepper;
//////////////////////////////////////////////////////////////////////////////////////////////
// analog read pin:
#define analogPin1  A1
#define analogPin2  A2
// digital wrte pin:
#define DIR_FLAG1  14
#define DIR_FLAG2  15


// Serial initialization

char receiveData = 'c';
String receiveData_upper = "";
String receiveData_side = "";
String receiveData_stepper = "";
String receiveData_test = "";

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
  pinMode(encoderPin_stepper_1, INPUT_PULLUP); 
  pinMode(encoderPin_stepper_2, INPUT_PULLUP);
  digitalWrite(encoderPin_upper_1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_upper_2, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_side_1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_side_2, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_stepper_1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin_stepper_2, HIGH); //turn pullup resistor on
  attachInterrupt(digitalPinToInterrupt(encoderPin_upper_1), updateEncoder_upper, CHANGE); 
  attachInterrupt(digitalPinToInterrupt(encoderPin_upper_2), updateEncoder_upper, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin_side_1), updateEncoder_side, CHANGE); 
  attachInterrupt(digitalPinToInterrupt(encoderPin_side_2), updateEncoder_side, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin_stepper_1), updateEncoder_stepper, CHANGE); 
  attachInterrupt(digitalPinToInterrupt(encoderPin_stepper_2), updateEncoder_stepper, CHANGE);

  //Setup for Motor
  //DCMotorSetup:
  pinMode(UpperMotEnable, OUTPUT);
  pinMode(UpperMotFwd, OUTPUT); 
  pinMode(UpperMotRev, OUTPUT); 
  TCCR1B = TCCR1B & 0b11111000 | 1;  // set 31KHz PWM to prevent motor noise
  upper_motor_PID.SetMode(AUTOMATIC);   //set PID in Auto mode
  upper_motor_PID.SetSampleTime(1);  // refresh rate of PID controller
  upper_motor_PID.SetOutputLimits(-100, 100); // this is the MAX PWM value to move motor, here change in value reflect change in speed of motor.
  pinMode(SideMotEnable, OUTPUT);
  pinMode(SideMotFwd, OUTPUT); 
  pinMode(SideMotRev, OUTPUT); 
  TCCR1B = TCCR1B & 0b11111000 | 1;  // set 31KHz PWM to prevent motor noise
  side_motor_PID.SetMode(AUTOMATIC);   //set PID in Auto mode
  side_motor_PID.SetSampleTime(1);  // refresh rate of PID controller
  side_motor_PID.SetOutputLimits(-100, 100); // this is the MAX PWM value to move motor, here change in value reflect change in speed of motor.
  //STEPPERMotorSetup:
  pinMode(DIR_FLAG1, OUTPUT);
  pinMode(DIR_FLAG2, OUTPUT);
  digitalWrite(DIR_FLAG1,LOW);
  digitalWrite(DIR_FLAG2,LOW);
  //wait for serial setup.
  delay(1000);
}

void loop() {
//  put your main code here, to run repeatedly:
////////////////////////////////////////////Read from Serial////////////////////////////////////////////////
  int seperate_flag = 0;
  int read_flag = 0;
  while(Serial.available()){ // only send data back if data has been sent
    delay(10);
    char c = Serial.read(); // read the incoming data
    receiveData_test += c;
//    Serial.print(c);
    
    if(c == ','){
      seperate_flag = 1;
    }
    else if(c == ';'){
      seperate_flag = 2;
    }
    else if(c == 'r'){
      read_flag = 1;
      c = 0;
    }
    else{
      if(seperate_flag == 0) receiveData_upper += c;
      else if(seperate_flag == 1) receiveData_side += c;
      else if(seperate_flag == 2) receiveData_stepper +=c;
    }
  }
  if (read_flag == 1){
//    Serial.println(receiveData_test);
//    Serial.print("(");
    Serial.println(encoderValue_upper);
//    Serial.print(",");
    Serial.println(encoderValue_side);
//    Serial.print(",");
    Serial.println(encoderValue_stepper);
//    Serial.println(")");
    receiveData = 'c'; // clear the flag comes into here
  }
  if (receiveData_upper.length() != 0 && receiveData_side.length() != 0 && receiveData_stepper.length() != 0){
    user_input_upper = receiveData_upper.toInt();
    user_input_side  = receiveData_side.toInt();
    user_input_stepper  = receiveData_stepper.toInt();
    clear_receiveData_string();
  }
  else{
    clear_receiveData_string();
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
  clear_receiveData_string(); //Is this necessary?

////////////////////////////for velocity control later !/////////////////////////////////////////////
//new velocity for input//
  delay(15);
  new_time = millis();
  long dif_time = new_time - old_time;
/////Averaging time///////////
  timeArray[indexArray] = dif_time;
  indexArray ++;
  if(indexArray > 9) {indexArray = 0;}
  long sum = 0;
  for (int i = 0; i < 10; i ++){
    sum = sum + timeArray[i];
  }
  long timeAverage = sum / 10;
//////////////////////////////
  new_position_upper = encoderValue_upper;
  new_position_side = encoderValue_side;
  int dif_upper = (new_position_upper - old_position_upper) * 100;
  int dif_side = (new_position_side - old_position_side) * 100;
  int vel_upper = dif_upper / timeAverage;
  int vel_side = dif_side / timeAverage;


//  Serial.print(REV_upper);
//  Serial.print(",");
//  Serial.print(vel_upper);
//  Serial.print(",");
//  Serial.print(encoderValue_upper);
//  Serial.print(",");
//  Serial.print(encoderValue_side);
//  Serial.print(",");
//  Serial.print(REV_side);
//  Serial.print(",");
//  Serial.print(vel_side);
//  Serial.print(",");
//  Serial.println(timeAverage);

  
  old_position_upper = new_position_upper;
  old_position_side = new_position_side;
  old_time = new_time;
/////////////////////////////////////////////////////////////////////////////////////////////////////
  
/////////////////////////////////////////////////////////////////////////////////////////////////////

  REV_upper = map (user_input_upper, -200, 200, -600, 600); // mapping degree into pulse
  REV_side = map (user_input_side, -200, 200, -600, 600); // mapping degree into pulse
  DIST_stepper = map(user_input_stepper, -500, 500, -400000, 400000); // 0.01 mm ----> encoder pulse
  if (REV_upper > 200){REV_upper = 200;}
  else if (REV_upper < -200){REV_upper = -200;}
  if (REV_side > 200){REV_side = 200;}
  else if (REV_side < -200){REV_side = -200;}
//////////////////////////////////////////DCMOTOR////////////////////////////////////////////////////
  //PID controller + PWM output
  //upper motor
  upper_setpoint = REV_upper;                  // set a new setpoint from python server
  upper_input = vel_upper;
  upper_motor_PID.Compute();                 // calculate new output
  if(encoderValue_upper < robot_boundary && encoderValue_upper > -robot_boundary){
    pwmOut(upper_output, UpperMotEnable);
    }
  else {pwmOut(0, UpperMotEnable);}
  //side motor
  side_setpoint = REV_side;                  // set a new setpoint from python server
  side_input = vel_side;
  side_motor_PID.Compute();                 // calculate new output
  if(encoderValue_side < robot_boundary && encoderValue_side > -robot_boundary){
    pwmOut(side_output, SideMotEnable);
  }
  else {pwmOut(0, SideMotEnable);}
//  for serial plot test

//  Serial.print(",");
//  Serial.print(DIST_stepper);
//  Serial.print(",");
//  Serial.println(encoderValue_stepper);
///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////STEPPER MOTOR///////////////////////////////////////////////
  stepper_motor_digitalwrite(DIST_stepper, encoderValue_stepper);
  receiveData_test = "";
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

void updateEncoder_stepper(){
  int MSB_stepper = digitalRead(encoderPin_stepper_1); //MSB = most significant bit
  int LSB_stepper = digitalRead(encoderPin_stepper_2); //LSB = least significant bit

  int encoded_stepper = (MSB_stepper << 1) |LSB_stepper; //converting the 2 pin value to single number
  int sum_stepper  = (lastEncoded_stepper << 2) | encoded_stepper; //adding it to the previous encoded value

  if(sum_stepper == 0b1101 || sum_stepper == 0b0100 || sum_stepper == 0b0010 || sum_stepper == 0b1011) encoderValue_stepper ++;
  if(sum_stepper == 0b1110 || sum_stepper == 0b0111 || sum_stepper == 0b0001 || sum_stepper == 0b1000) encoderValue_stepper --;

  lastEncoded_stepper = encoded_stepper; //store this value for next time

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

void clear_receiveData_string(){
    receiveData_upper = "";
    receiveData_side = "";
    receiveData_stepper = "";
}

void stepper_motor_digitalwrite(long DIST_stepper, long encoderValue_stepper){
    if (DIST_stepper + 20 < encoderValue_stepper){
    digitalWrite(DIR_FLAG1,HIGH);
    digitalWrite(DIR_FLAG2,LOW);
  }
  else if(DIST_stepper - 20 > encoderValue_stepper){
    digitalWrite(DIR_FLAG1,LOW);
    digitalWrite(DIR_FLAG2,HIGH);
  }
  else{
    digitalWrite(DIR_FLAG1,LOW);
    digitalWrite(DIR_FLAG2,LOW);
  }
}
