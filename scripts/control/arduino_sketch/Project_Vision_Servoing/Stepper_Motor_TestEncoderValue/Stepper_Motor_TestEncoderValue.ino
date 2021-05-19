/*     Simple Stepper Motor Control Exaple Code
 *      
 *  by Dejan Nedelkovski, www.HowToMechatronics.com
 *  
 */
// defines pins numbers
#define EN_PIN    7  // Nano v3:  16 Mega:  38  UNO:  7 //enable (CFG6)
#define DIR_PIN   8  //           19        55        8 //direction
#define STEP_PIN  9  //           18        54        9 //step
#define CS_PIN    10 //           17        64        10//chip select

bool dir = true;


int encoderPin1 = 2; //Encoder Output 'A' must connected with intreput pin of arduino.
int encoderPin2 = 3; //Encoder Otput 'B' must connected with intreput pin of arduino.
volatile int lastEncoded = 0; // Here updated value of encoder store.
volatile int encoderValue = 0; // Raw encoder value
int PPR = 2;  // Encoder Pulse per revolution.
int lastMSB = 0;
int lastLSB = 0;

//Linear Stage 16//05   :16 mm in outer diameter//05 mm per revolution
//Microstepping 16---> 360 / 1.8 * 16 = 3200 steps per revolution
//                     5 mm/ 3200 steps == 0.0015625 mm/step
int microstepping = 64;
int one_revolution = 200 * microstepping / 5;

const int motorSpeedLinearStage = 30;
int lastReceive = 0;
int counter = 0;


#include <TMC2130Stepper.h>
TMC2130Stepper TMC2130 = TMC2130Stepper(EN_PIN, DIR_PIN, STEP_PIN, CS_PIN);



void setup() {
  // Sets the two pins as Outputs
  pinMode(EN_PIN,OUTPUT); 
  pinMode(DIR_PIN,OUTPUT);
  digitalWrite(DIR_PIN,HIGH); // Enables the motor to move in a particular direction
  digitalWrite(EN_PIN,LOW);

  // Serial Setup:
  Serial.begin(115200); // set the baud rate
  //Interrupt Setup:
  pinMode(encoderPin1, INPUT_PULLUP); 
  pinMode(encoderPin2, INPUT_PULLUP);
  digitalWrite(encoderPin1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin2, HIGH); //turn pullup resistor on
  attachInterrupt(0, updateEncoder, CHANGE); 
  attachInterrupt(1, updateEncoder, CHANGE);

  //TMC2130 SETUP
  TMC2130.begin(); // Initiate pins and registeries
  TMC2130.SilentStepStick2130(900); // Set stepper current to 600mA
  TMC2130.stealthChop(1); // Enable extremely quiet stepping
  TMC2130.microsteps(microstepping);
  digitalWrite(EN_PIN, LOW);
  
  //delay
  delay(100);
  TMC2130.shaft_dir(true);    
}
void loop() {
  
//Receive char(s) from python serial communication
  counter = counter + 1;
  digitalWrite(STEP_PIN,HIGH);
  delayMicroseconds(30);
  digitalWrite(STEP_PIN,LOW);
  delayMicroseconds(30);

  if(encoderValue == 3000 || encoderValue == -3000){
    dir = !dir;
    TMC2130.shaft_dir(dir);
    delay(100);
  }
  
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
