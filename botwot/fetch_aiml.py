# Python script to scrape and fetch all aiml files
import requests
import BeautifulSoup as bs
import time
import xml.etree.cElementTree as et
import os


def fetch_aiml_files():
    """
    Function to fetch all the available aiml files
    :return: None
    """
    start_time = time.time()
    # Creating dir for aiml files
    if not os.path.exists("./aiml_files"):
        os.makedirs("./aiml_files")
    base_url = "http://www.alicebot.org/aiml/aaa/"

    request_data = requests.get(base_url)
    bs_obj = bs.BeautifulSoup(request_data.content)

    for link in bs_obj.findAll('a'):
        file_name = link.get('href')
        if ".aiml" in file_name:
            f = open("./aiml_files/"+str(file_name), 'w')
            url = str(base_url)+str(file_name)
            print url
            r = requests.get(url)
            f.write(r.content)
            f.close()
            print "File {} done".format(str(file_name))
        else:
            continue

    print "Function complete in {} seconds".format(time.time() - start_time)


def edit_startup_file():
    """
    Function to create and edit the startup file which is used for loading the aiml.
    :return: None
    """
    root = et.Element("aiml",version="1.0.1",encoding="UTF-8")
    category = et.SubElement(root, "category")
    # et.SubElement(category, "pattern").text = "LOAD AIML B"
    template = et.SubElement(category, "template")
    files = os.listdir("./aiml_files/")
    for file_name in files:
        et.SubElement(template, "learn").text = "aiml_files/"+str(file_name)
    tree = et.ElementTree(root)
    tree.write("std-startup.xml")


if __name__ == "__main__":
    edit_startup_file()