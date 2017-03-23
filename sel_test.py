import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains


'''profile = webdriver.ChromeOptions()
profile.add_argument('webdriver.load.strategy')'''
driver = webdriver.Chrome('C:\Python27\chromedriver.exe')
try:
    driver.set_page_load_timeout(6)
    driver.get('https://hh.ru/')
except TimeoutException as ex:
    isrunning = 0
    ActionChains(driver).send_keys(Keys.CONTROL, "esc").perform()
    print("Exception has been thrown. " + str(ex))
    

'''try:
  driver.get('https://hh.ru/')
except TimeoutException:
  driver.execute_script('window.stop()')'''
'''driver.get('https://hh.ru/')'''

'''all_cookies = driver.get_cookies()
driver.delete_all_cookies()
for cookie in all_cookies:
    driver.add_cookie(cookie)'''
button=driver.find_element_by_xpath("/html/body/div[5]/div/div[3]/div/div/div[2]/p/span[4]/a")
button.click()
user_name = driver.find_element_by_name("email")
user_name.send_keys("******")
password = driver.find_element_by_name("*****")
password.send_keys("xxborn2048kill11")
driver.find_element_by_class_name('button').click()
driver.get('https://hh.ru/resume/5c3d2133ff039903370039ed1f4b77444b4176')
resume_update=driver.find_element_by_xpath("/html/body/div[5]/div[2]/div/div/div/div[4]/div[2]/div/div[2]/div/div/div/div[2]/div[3]/div[2]/button")
resume_update.click()

'''pickle.dump( driver.get_cookies() , open("cookies.pkl","wb"))'''
'''cookies = pickle.load(open("cookies.pkl", "rb"))'''

'''driver.close()
driver = webdriver.Chrome('C:\Python27\chromedriver.exe')

driver.delete_all_cookies()#выпилю старые прежде чем загрузить новые
for cookie in cookies:
    driver.add_cookie(cookie)
driver.get('https://oauth.vk.com/authorize?display=mobile&response_type=code&client_id=3295164&redirect_uri=https%3A%2F%2Fhhid.ru%2Foauth2%2Fcode&scope=4194304&state=token%3DUW9Rqq8c27f4HP8r4ohifMFNfXCWlVb5KZ_lMvMaJL3Uo0IkuYM2bg7nI8c8Cpei555sEelqaCxn40BKHND3937BdYYB%26reg%3Dhttps%253A%252F%252Fhh.ru%252Faccount%252Fconnect%252Fregister%26fail%3Dhttps%253A%252F%252Fhh.ru%252Faccount%252Fconnect%252Fresult%253Ffail%253Dtrue%26login%3Dhttps%253A%252F%252Fhh.ru%252Faccount%252Fconnect%252Fresult%26system%3DVK%26mergeOAuth%3Dfalse')

assert "IT" in driver.title
elem = driver.find_element_by_name("q")
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
assert "No results found." not in driver.page_source'''
'''driver.close()'''
