#include<LiquidCrystal_I2C.h>
#include<SoftwareSerial.h>
LiquidCrystal_I2C lcd(0x27,16,2);
 
SoftwareSerial ss(2,3);
void setup() {
 
Serial.begin(9600);
ss.begin(9600);
lcd.init();
lcd.init();
lcd.backlight();
lcd.setCursor(0,0);
lcd.print(" BP Sensor V3");
lcd.setCursor(0,1);
lcd.print(" www.robosap.in");
delay(3000);
lcd.clear();
 
}
 
void loop() {
 
if(ss.available()>0)
{
 
String instr=ss.readStringUntil('\n');
if(instr.indexOf(',')>0)
{
int findex=instr.indexOf(',');
 
String bpdata = instr.substring(findex+1);
 
bpdata.trim();
 
Serial.println(bpdata);

 
}
else
{
instr.trim();
Serial.println(instr);

 
}
}
}