// DC motor gear ratio : 200 
// Encoder 2 pulse per revolution

String receiveData;

//This is the section for encoder and ISR
int encoderPin1 = 2; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin2 = 3; //Encoder Otput 'B' must connected with intreput pin of arduino.
volatile int lastEncoded = 0; // Here updated value of encoder store.
volatile int encoderValue = 0; // Raw encoder value
int PPR = 2;  // Encoder Pulse per revolution.
int lastMSB = 0;
int lastLSB = 0;

int angle = 360; // Maximum degree of motion.
int REV = 0;          // Set point REQUIRED ENCODER VALUE


//Thsi is the section for Motor
#include <PID_v1.h>
#define MotEnable 9 //Motor Enamble pin Runs on PWM signal
#define MotFwd  6  // Motor Forward pin
#define MotRev  7 // Motor Reverse pin
int motor_limit_motion = 800;

//PID controller setup
//double kp = 0.8 , ki = 0.0001 , kd = 0.0012;
//double kp = 0.87 , ki = 0.0002 , kd = 0.0015;
double kp = 5 , ki = 40 , kd = 0.002;

double input = 0, output = 0, setpoint = 0;
PID myPID(&input, &output, &setpoint, kp, ki, kd, DIRECT);


void setup() {
  // put your setup code here, to run once:
  
  //MotorSetup:
  pinMode(MotEnable, OUTPUT);
  pinMode(MotFwd, OUTPUT); 
  pinMode(MotRev, OUTPUT); 
  TCCR1B = TCCR1B & 0b11111000 | 1;  // set 31KHz PWM to prevent motor noise
  myPID.SetMode(AUTOMATIC);   //set PID in Auto mode
  myPID.SetSampleTime(1);  // refresh rate of PID controller
  myPID.SetOutputLimits(-150, 150); // this is the MAX PWM value to move motor, here change in value reflect change in speed of motor.
 
  //Seup for Serial connection:
  Serial.begin(115200); // set the baud rate
  Serial.println(5); // print "Ready" once
  receiveData   = "0";
  Serial.flush();
  
  //Setup: interrupts for motors and encoders
  pinMode(encoderPin1, INPUT_PULLUP); 
  pinMode(encoderPin2, INPUT_PULLUP);
  digitalWrite(encoderPin1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin2, HIGH); //turn pullup resistor on
  attachInterrupt(0, updateEncoder, CHANGE); 
  attachInterrupt(1, updateEncoder, CHANGE);
  delay(100);
}

void loop() {
  //This first section is for receiving data and sending data
  //Receive char(s) from python serial communication
  Serial.flush();
  while(Serial.available()){ // only send data back if data has been sent
    delay(3);
    char c = Serial.read(); // read the incoming data
    receiveData += c;
  }
  if(receiveData.length()>0){ //Verify that the variable contains information
    int User_Input = receiveData.toInt(); //store the data from serial input in interger type
//    receiveData = ""; //clear out the store data // put it into pwmOut function.
    delay(3);
    REV = map (User_Input, -90, 90, -2400, 2400 ); // mapping degree into pulse
  }
  //This second section is for motor setup
//  Serial.print("this is REV - "); 
  Serial.println(encoderValue);
  if(REV < 2400 && REV > -2400){
  setpoint = REV;
  }
  input = encoderValue;
  myPID.Compute();                 // calculate new output
  pwmOut(output);
    
  delay(40); // delay for 1/10 of a second
}
void updateEncoder(){
  int MSB = digitalRead(encoderPin1); //MSB = most significant bit
  int LSB = digitalRead(encoderPin2); //LSB = least significant bit

  int encoded = (MSB << 1) |LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue --;

  lastEncoded = encoded; //store this value for next time

}

void pwmOut(int out) {                               
  if (out > 0) {                         // if REV > encoderValue motor move in forward direction.    
    analogWrite(MotEnable, out);         // Enabling motor enable pin to reach the desire angle
    forward();                           // calling motor to move forward
  }
  else {
    analogWrite(MotEnable, abs(out));          // if REV < encoderValue motor move in forward direction.                      
    reverse();                            // calling motor to move reverse
  }
  receiveData=""; // Cleaning User input, ready for new Input
}

void forward () {
    digitalWrite(MotFwd, HIGH); 
    digitalWrite(MotRev, LOW); 
}

void reverse () {
  digitalWrite(MotFwd, LOW); 
  digitalWrite(MotRev, HIGH); 
  
}
void finish () {
  digitalWrite(MotFwd, LOW); 
  digitalWrite(MotRev, LOW); 
  
}
