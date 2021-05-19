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
#define DIR_FLAG1 2
#define DIR_FLAG2 4

//Linear Stage  16//05   :16 mm in outer diameter//05 mm per revolution
//Microstepping 16 ---> 360 / 1.8 * 16 = 3200 steps per revolution
//                     5 mm/ 3200 steps == 0.0015625 mm/step

int microstepping = 64;
int one_revolution = 200 * microstepping;
const int motorSpeedLinearStage = 50;
#include <TMC2130Stepper.h>
TMC2130Stepper TMC2130 = TMC2130Stepper(EN_PIN, DIR_PIN, STEP_PIN, CS_PIN);

void setup() {
  // Sets the two pins as Outputs
  pinMode(EN_PIN,OUTPUT); 
  pinMode(DIR_PIN,OUTPUT);
  digitalWrite(DIR_PIN,HIGH); // Enables the motor to move in a particular direction
  digitalWrite(EN_PIN,LOW);

  // SETs two pins as Inputs
  pinMode(DIR_FLAG1, INPUT);
  pinMode(DIR_FLAG2, INPUT);

  //TMC2130 SETUP
  TMC2130.begin(); // Initiate pins and registeries
  TMC2130.SilentStepStick2130(900); // Set stepper current to 600mA
  TMC2130.stealthChop(1); // Enable extremely quiet stepping
  TMC2130.microsteps(microstepping);
  digitalWrite(EN_PIN, LOW);

  //testing
  Serial.begin(115200);
    
}
void loop() {

if(digitalRead(DIR_FLAG1) == HIGH && digitalRead(DIR_FLAG2) == LOW){
  actuate_forward();
  Serial.println("f");
  }
else if(digitalRead(DIR_FLAG1) == LOW && digitalRead(DIR_FLAG2) == HIGH){
  actuate_backward();
  Serial.println("b");

  }
else{
  digitalWrite(EN_PIN, LOW);
  Serial.println("0");
  }
//
//  if(digitalRead(DIR_FLAG1) == HIGH && digitalRead(DIR_FLAG2) == LOW ){
//    Serial.println("10");
//  }
//  else if(digitalRead(DIR_FLAG1) == LOW && digitalRead(DIR_FLAG2) == HIGH){
//    Serial.println("01");
//  }
//  else if(digitalRead(DIR_FLAG1) == HIGH && digitalRead(DIR_FLAG2) == HIGH){
//    Serial.println("11");
//  }
//  else if(digitalRead(DIR_FLAG1) == LOW && digitalRead(DIR_FLAG2) == LOW){
//    Serial.println("00");
//  }


}

void actuate_forward(){
  digitalWrite(EN_PIN, LOW);   //Enabling driver
  digitalWrite(DIR_PIN, HIGH); //Changes the rotations direction
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(motorSpeedLinearStage);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(motorSpeedLinearStage);
}

void actuate_backward(){
  digitalWrite(EN_PIN, LOW);   //Enabling driver
  digitalWrite(DIR_PIN, LOW); //Changes the rotations direction
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(motorSpeedLinearStage);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(motorSpeedLinearStage);
}
