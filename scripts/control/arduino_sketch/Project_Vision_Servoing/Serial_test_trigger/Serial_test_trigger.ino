// DC motor gear ratio : 200 
// Encoder 2 pulse per revolution

String receiveData;
String abc = "abc";
String def = "def";

int counter = 33;

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


void setup() {
  // put your setup code here, to run once:

  pinMode(LED_BUILTIN, OUTPUT);


  //Seup for Serial connection:
  Serial.begin(115200); // set the baud rate
  //Serial.println(5); // print "Ready" once
  //receiveData   = "0";
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
  counter = counter + 1;
  delay(10);
  while(Serial.available()){ // only send data back if data has been sent
    delay(3);
    char c = Serial.read(); // read the incoming data
    receiveData += c;
  }  
  if(receiveData.length()>0){ //Verify that the variable contains information
    if(receiveData == abc){
      Serial.println(counter);
      delay(5);
    }
    else if(receiveData == def){
      for (int i=0; i<10; i++){
      digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
      delay(20);                       // wait for a second
      digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
      delay(20); 
      }
    }
    Serial.flush();
  }
  
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  
  //Clear Data:
  receiveData = "";
  if(counter > 2048){
    counter = 33;
  }
  Serial.flush();
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
