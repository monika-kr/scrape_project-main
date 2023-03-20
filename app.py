import io
import json
import pandas as pd
import re
import requests, json, lxml

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from bs4 import BeautifulSoup
import requests
import urllib
from PIL import Image
import requests
from io import BytesIO

from flask import render_template, session, redirect
from flask import Flask

app = Flask(__name__, template_folder='template')

@app.route('/', methods=("POST", "GET"))
def html_table():
#@app.route('/')
#def success():

    item_name = []
    prices = []
    links = []
    location = []
    prediction = []

    for i in range(1,2):

        ebayUrl = "https://www.ebay.co.uk/sch/i.html?_nkw=wooden+pallets&_sop=12&_pgn="+str(i)
        html_page = requests.get(ebayUrl)
        soup = BeautifulSoup(html_page.content, "html.parser")
        container = soup.find('div', class_="srp-river srp-layout-inner")

        listings = soup.find_all('li', attrs={'class': 's-item'})

        for j in range(1,len(listings)):
            images = container.find_all('img')
            example = images[j]
            a = example.attrs['src']
            response = requests.get(a)
            img = Image.open(BytesIO(response.content))
          #  prod_pred = predict_image(img)
          #  prediction.append(prod_pred)
        # print(prod_pred)

        for listing in listings:
            prod_price = " "
            prod_link = " "
            prod_pred = " "

            for price in listing.find_all('span', attrs={'class':"s-item__price"}):
                    prod_price = str(price.find(text=True, recursive=False))
                #    prod_price = int(re.sub(",","",prod_price.split("INR")[1].split(".")[0]))
                    prices.append(prod_price)
                    
            for link in listing.find_all('a', attrs={'class':"s-item__link"}, href=True):
                    prod_link = link['href']
                    prod_link_clean = prod_link.split("?")[0]
                    links.append(prod_link_clean)

            for title in soup.select(".s-item__title span"):
                if "Shop on eBay" in title:
                    pass
                else:
                    item_name.append(title.text)
                
    ebay_model = torch.load('ebay_model.pth')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

  #  soup = BeautifulSoup(html_page.content, "html.parser")
   # container = soup.find('div', class_="srp-river srp-layout-inner")
   # images = container.find_all('img')
   # prediction = []

    def predict_image(image):
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = ebay_model(input)
        index = output.data.cpu().numpy().argmax()
        return index

    for j in range(1,len(images)):
        prod_pred = " "
        images = container.find_all('img')   
        example = images[j]
        a = example.attrs['src']
        response = requests.get(a)
        img = Image.open(BytesIO(response.content))
        prod_pred = predict_image(img)
        prediction.append(prod_pred)
        #print(prod_pred)                        

    df = pd.DataFrame({"Listing title": item_name[1:len(prices)], "Prices": prices[1:], "url": links[1:], "Prediction": prediction})
    #df = pd.DataFrame({"Listing title": item_name[1:len(prices)], "Prices": prices[1:], "url": links[1:]})
    df = df.iloc[1: , :]

    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values, formatters={'Link':lambda x:f'<a href="{x}">{x}</a>'})

 
if __name__ == '__main__':
    app.run(debug=True)