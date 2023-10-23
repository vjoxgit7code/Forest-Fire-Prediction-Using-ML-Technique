import streamlit as st
import pickle
import requests
import datetime
import json
from geopy.geocoders import Nominatim
import math
import time
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import codecs
import streamlit.components.v1 as stc 
import openai
from streamlit_chat import message


import requests
from geopy.geocoders import Nominatim
import logging

def weather(zip_code, country_code):
    try:
        api_key = "f10eff398a3485f33ff8ab51ef3bab2f"
        base_url = "https://api.openweathermap.org/data/2.5/weather?zip="
        complete_url = base_url + str(zip_code) + ',' + country_code.upper() + '&appid=' + api_key
        response = requests.get(complete_url)
        response.raise_for_status() # raises an HTTPError if the response status is not 200
        weather_data = response.json()
        # latitude
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(zip_code)

        latitude = location.latitude
        weather_data['latitude'] = latitude
        print('latitude:', latitude)

        # longitude
        longitude = location.longitude
        weather_data['longitude'] = longitude
        print('longitude:', longitude)

        # location Name
        current_temperature = round(weather_data['main']['temp'] - 273.15)
        humidity=weather_data['main']['humidity']
        windspeed=weather_data['wind']['speed']
        print('Current Temperature:', current_temperature)

        feels_like = weather_data['main']['feels_like'] - 273.15
        print('feels_like:', feels_like)

        print('Pressure:', weather_data['main']['pressure'])
        print('humidity:', weather_data['main']['humidity'])
        print('WS:', weather_data['wind']['speed'])
        print('0' * 100)

        return latitude, current_temperature, humidity, windspeed, longitude
    except (requests.exceptions.RequestException, KeyError, AttributeError, TypeError, ValueError,ConnectionError) as e:
        logging.error(f"Error fetching weather data for {zip_code}: {e}")
        return None


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_ynjalQelI1.json"
lottie_url_download = "https://assets8.lottiefiles.com/packages/lf20_ynjalQelI1.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)

#Home page--------------------------------------------------------------------------------------------------------------------------------
def homepage(new_html,width=1000,height=1850):
    calc_file=codecs.open(new_html,'r')
    page=calc_file.read()
    stc.html(page,width=width,height=height,scrolling=False )


#Fire Vulnerability-----------------------------------------------------------------------------------------------------------------------

class InvalidLatitude(Exception):
    """Exception to handle variables not covered by DataDict"""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value) + " is not a valid Latitude."


def FFMC(TEMP,RH,WIND,RAIN,FFMCPrev):


    RH = min(100.0,RH)
    mo = 147.2 * (101.0 - FFMCPrev) / (59.5 + FFMCPrev)

    if RAIN > .5:
        rf = RAIN - .5

        if mo <= 150.0:
            mr = mo + \
                 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0-math.exp(-6.93 / rf))
        else:

            mr = mo + \
                 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0-math.exp(-6.93 / rf)) + \
                 0.0015 * pow(mo - 150.0, 2) * pow(rf, .5)

        if mr > 250.0:
            mr = 250.0

        mo=mr

    ed = 0.942 * pow(RH, 0.679) + \
         11.0 * math.exp((RH - 100.0) / 10.0) + 0.18 * (21.1 - TEMP) * (1.0 - math.exp(-0.115 * RH))

    if mo > ed:
        ko = 0.424 * (1.0 - pow(RH / 100.0, 1.7)) + \
             0.0694 * pow(WIND, .5) * (1.0 - pow(RH / 100.0, 8))

        kd = ko * 0.581 * math.exp(0.0365 * TEMP)

        m = ed + (mo - ed) * pow(10.0,-kd)

    else:
        ew = 0.618 * pow(RH,0.753) + \
             10.0 * math.exp((RH - 100.0) / 10.0) + \
             0.18 * (21.1 - TEMP) * (1.0 - math.exp(-0.115 * RH))
        if mo < ew:
            k1 = 0.424 * (1.0 - pow((100.0 - RH) / 100.0, 1.7)) + \
                 0.0694 * pow(WIND, .5) * (1.0 - pow((100.0 - RH) / 100.0, 8))

            kw = k1 * 0.581 * math.exp(0.0365 * TEMP)

            m = ew - (ew - mo) * pow(10.0, -kw)
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)



def DMC(TEMP,RH,RAIN,DMCPrev,LAT,MONTH):


    RH = min(100.0,RH)
    if RAIN > 1.5:
        re = 0.92 * RAIN - 1.27

        mo = 20.0 + math.exp(5.6348 - DMCPrev / 43.43)

        if DMCPrev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * DMCPrev)
        else:
            if DMCPrev <= 65.0:
                b = 14.0 - 1.3 * math.log(DMCPrev)
            else:
                b = 6.2 * math.log(DMCPrev) - 17.2
        
        mr = mo + 1000.0 * re / (48.77 + b * re)

        pr = 244.72 - 43.43 * math.log(mr - 20.0)

        if pr > 0.0:
            DMCPrev = pr
        else:
            DMCPrev = 0.0

    if TEMP > -1.1:
        d1 = DayLength(LAT,MONTH)

        k = 1.894 * (TEMP + 1.1) * (100.0 - RH) * d1 * 0.000001

    else:
        k = 0.0

    return DMCPrev + 100.0 * k



def DC(TEMP,RAIN,DCPrev,LAT,MONTH):


    if RAIN > 2.8:
        rd = 0.83 * RAIN - 1.27
        Qo = 800.0 * math.exp(-DCPrev / 400.0)
        Qr = Qo + 3.937 * rd
        Dr = 400.0 * math.log(800.0 / Qr)
        
        if Dr > 0.0:
            DCPrev = Dr
        else:
            DCPrev = 0.0

    Lf = DryingFactor(LAT, MONTH-1)

    if TEMP > -2.8:
        V = 0.36 * (TEMP+2.8) + Lf
    else:
        V = Lf
    
    if V < 0.0:
        V = 0.0

    D = DCPrev + 0.5 * V

    return D



def ISI(WIND,FFMC):

    fWIND = math.exp(0.05039 * WIND)

    m = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)

    fF = 91.9 * math.exp(-0.1386 * m) * (1.0 + pow(m, 5.31) / 49300000.0)

    return 0.208 * fWIND * fF



def BUI(DMC,DC):

    if DMC <= 0.4 * DC:
        U = 0.8 * DMC * DC / (DMC + 0.4 * DC)
    else:
        U = DMC - (1.0 - 0.8 * DC / (DMC + 0.4 * DC)) * \
            (0.92 + pow(0.0114 * DMC, 1.7))

    return max(U,0.0)



def FWI(ISI, BUI):

    if BUI <= 80.0:
        fD = 0.626 * pow(BUI, 0.809) + 2.0
    else:
        fD = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * BUI))

    B = 0.1 * ISI * fD

    if B > 1.0:
        S = math.exp(2.72 * pow(0.434 * math.log(B), 0.647))
    else:
        S = B

    return S



def DryingFactor(Latitude, Month):

    LfN = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    LfS = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]

    if Latitude > 0:
        retVal = LfN[Month]
    elif Latitude <= 0.0:
        retVal = LfS[Month]

    return retVal



def DayLength(Latitude, MONTH):
    '''Approximates the length of the day given month and latitude'''

    DayLength46N = [ 6.5,  7.5,  9.0, 12.8, 13.9, 13.9, 12.4, 10.9,  9.4,  8.0,  7.0,  6.0]
    DayLength20N = [ 7.9,  8.4,  8.9,  9.5,  9.9, 10.2, 10.1,  9.7,  9.1,  8.6,  8.1,  7.8]
    DayLength20S = [10.1,  9.6,  9.1,  8.5,  8.1,  7.8,  7.9,  8.3,  8.9,  9.4,  9.9, 10.2]
    DayLength40S = [11.5, 10.5,  9.2,  7.9,  6.8,  6.2,  6.5,  7.4,  8.7, 10.0, 11.2, 11.8]

    retVal = None

    if Latitude<= 90 and Latitude > 33:
        retVal = DayLength46N[MONTH-1]
    elif Latitude <= 33 and Latitude > 0.0:
        retVal = DayLength20N[MONTH-1]
    elif Latitude <= 0.0 and Latitude > -30.0:
        retVal = DayLength20S[MONTH-1]
    elif Latitude <= -30.0 and Latitude >= -90.0:
        retVal = DayLength40S[MONTH-1]

    if retVal==None:
        raise InvalidLatitude(Latitude)

    return retVal

def calcFWI(MONTH,TEMP,RH,WIND,RAIN,FFMCPrev,DMCPrev,DCPrev,LAT):

    ffmc = FFMC(TEMP,RH,WIND,RAIN,FFMCPrev)
    dmc = DMC(TEMP,RH,RAIN,DMCPrev,LAT,MONTH)
    dc = DC(TEMP,RAIN,DCPrev,LAT,MONTH)
    isi = ISI(WIND, ffmc)
    bui = BUI(dmc,dc)
    fwi = FWI(isi, bui)
   


    return (ffmc,dmc,dc,isi,bui,fwi)

#Blog-----------------------------------------------------------------------------------------------------------------------------------------
def blog(new_html,width=1200,height=900):
    calc_file=codecs.open(new_html,'r')
    page=calc_file.read()
    stc.html(page,width=width,height=height,scrolling=False )
  

#Model--------------------------------------------------------------------------------------------------------------------------------------
model2=pickle.load(open(r"Randomforest.pkl",'rb'))
#chatbot-----------------------------------------------------------------------------------------------------------------------------------
#navigation box--------------------------------------------------------------------------------------------------------------------------
def navigation(st):
    Menu=['Home','Weather','Fire vulnerability','Upload a Fire','Prevention Techniques']
    choice=st.sidebar.selectbox('Main Menu',Menu)
    if choice=='Home':
        homepage('new.html')
              
    elif choice=='Weather':
        st.title('Weather Finder')
        if __name__ == '__main__':
            zip_code = st.text_input("Enter zip code:",max_chars=6)
            
            btn=st.button('ENTER')
                
            
            if btn==True:
                with st_lottie_spinner(lottie_download,speed=2,reverse=True,loop=True,quality='high',height=180,width=100,key='Enter'):
                    time.sleep(5)

                    
                    
                if zip_code :
                    try:
                        latitude,current_temperature, humidity, windspeed,longitude= weather(zip_code, country_code='in')
                        st.write("Latitude:", latitude)
                        st.write('Longitude:',longitude)
                        st.write("Current Temperature:", current_temperature,'C')
                        st.write("Humidity:", humidity)
                        st.write("Wind Speed:", windspeed,'Km/h')
                        weather(zip_code,country_code='in')
                    except (requests.exceptions.RequestException, KeyError, AttributeError, TypeError, ValueError,ConnectionError) as e:
                        logging.error(f"Error fetching weather data for {zip_code}: {e}")
                        return st.write('Check your code and try again:')    
    elif choice=='Fire vulnerability':
        
        st.title('Fire Predictor')
        if __name__ == '__main__':
            zip_code = st.text_input("Enter zip code:",max_chars=6)
            
            
            btn=st.button('Predict')
        if btn==True:
            with st_lottie_spinner(lottie_download,speed=2,reverse=True,loop=True,quality='high',height=180,width=100,key='Enter'):
                    time.sleep(5)
                    
            if zip_code:
                try:
                    latitude,current_temperature, humidity, windspeed,longitude=weather(zip_code,country_code='in')
                    current_date_and_time = datetime.datetime.now()
                    Month = current_date_and_time.month
                    day = current_date_and_time.weekday()
                    year = current_date_and_time.today().year
                    rain = 0
                    FFMCPrev = 83.62
                    DMCPrev = 14.008974358974356
                    DCPrev = 48.47621794871795
                                           
                    Lat=latitude
                    temperature=current_temperature
                    ffmc, dmc, dc, isi, bui, fwi = calcFWI(Month,temperature,humidity,windspeed,rain,FFMCPrev,DMCPrev,DCPrev,Lat)
                    X1=[[day,Month,year,temperature,humidity,windspeed,rain,ffmc,dmc,dc,isi,bui,fwi]]
                    print(model2.predict(X1))
                

                
                    if model2.predict(X1)==1:
                        st.write( 'You Are Not in Fire Zone ')
                        st.write('If Any problem persists contact the Helpline:')

                                    
                    else:
                        st.write('You are in fire zone !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        st.write('Help_line=7994241228')
                        st.write("Fire Help:",
                        "https://fsi.nic.in/",
                        unsafe_allow_html=True)

                
                except (requests.exceptions.RequestException, KeyError, AttributeError, TypeError, ValueError,ConnectionError) as e:
                    logging.error(f"Error fetching weather data for {zip_code}: {e}")
                    return st.write('Check your code and try again:')

   
    elif choice == 'Upload a Fire':
        st.title('Upload A Fire')
        st.write('Be serious, We will be very quick.')
        st.markdown("---")
        image = st.file_uploader('Your file should be in an appropriate format', type=['jpg', 'png', 'svg'])
        if image is not None:
            st.image(image)
            contact = st.text_input('Enter your contact details:')
        btn = st.button("Upload")
        if btn:
            st.write('We will contact you within 2 minutes.')
            print('Details:',contact)
    
    elif choice == 'Prevention Techniques':
        blog('blog.html')
    

        
        

        


                         
        


    


navigation(st)
hide_streamlit_style="""
<style>
#MainMenu{visibility:hidden}
footer{visibility:hidden}
<style>
"""
st.markdown(hide_streamlit_style,unsafe_allow_html=True)
