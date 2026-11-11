/* ESP32 interfacing with ADS1115 */

/****************************
Channel 2
  when nothing is plugged in it is at 4649 (normally it would be around 26290)
  when something is plugged in, it does not react -- but instead, Channel 1 reacts (!)

Channel 3
  plug is very hard to put in 

****************************/
#include<ADS1115_lite.h>
ADS1115_lite ads(ADS1115_DEFAULT_ADDRESS);  // 0x48 addr pin connected to GND
float lp_read_a0 = 0.0;
float lp_read_a1 = 0.0;
float lp_read_a2 = 0.0;
float lp_read_a3 = 0.0;


void setup(){
 Serial.begin(115200);
  delay(10);
  Wire.begin(21, 22);  // SDA=D21, SCL=D22
  ads_config();


}

void loop()
{
  float alpha = 0.0;

  lp_read_a0 = alpha * lp_read_a0 + (1.0 - alpha) * ((float) ads_read(0));
  lp_read_a1 = alpha * lp_read_a1 + (1.0 - alpha) * ((float) ads_read(1));
  lp_read_a2 = alpha * lp_read_a2 + (1.0 - alpha) * ((float) ads_read(2));
  lp_read_a3 = alpha * lp_read_a3 + (1.0 - alpha) * ((float) ads_read(3));

  
  Serial.print((int) lp_read_a0);
  Serial.print(";");
  Serial.print((int)lp_read_a1);
  Serial.print(";");
  Serial.print((int) lp_read_a2);
  Serial.print(";");
  Serial.println((int) lp_read_a3);
}

void ads_config()
{
  ads.setGain(ADS1115_REG_CONFIG_PGA_4_096V);
  ads.setSampleRate(ADS1115_REG_CONFIG_DR_16SPS);
}

int16_t ads_read(uint8_t channel)
{ 
  if (channel == 0) 
  {
    ads.setMux(ADS1115_REG_CONFIG_MUX_SINGLE_0);  // AIN0 - GND
  } 
  else if (channel == 1) 
  {
    ads.setMux(ADS1115_REG_CONFIG_MUX_SINGLE_1);  // AIN1 - GND
  }
else if (channel == 2) 
  {
    ads.setMux(ADS1115_REG_CONFIG_MUX_SINGLE_2);  // AIN2 - GND
  }
else 
  {
    ads.setMux(ADS1115_REG_CONFIG_MUX_SINGLE_3);  // AIN3 - GND
  }
 ads.triggerConversion();
  return ads.getConversion();
}