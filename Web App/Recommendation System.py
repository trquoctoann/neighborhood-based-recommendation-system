import os 
import pandas as pd 
import numpy as np
from pathlib import Path 
import flask
import datetime as dt

# read data, just change the path to the place where you store this repo
path = 'D:\Giáo Trình\Kỳ 5\Kho và Khai Phá Dữ Liệu\Project'
customerListId = pd.read_csv(os.path.join(path, 'Web App\\Data\\customer_id.csv'))
productListId = pd.read_csv(os.path.join(path, 'Web App\\Data\\product_id.csv'))
prodRules = pd.read_csv(os.path.join(path, 'Web App\\Data\\prod_rules.csv'))
customerSegment = pd.read_csv(os.path.join(path, 'Web App\\Data\\rfm.csv'))
recommendData = pd.read_csv(os.path.join(path, 'Web App\\Data\\recommendData.csv'))
rawRating = Path(os.path.join(path, 'Web App\\Data\\spare_matrix.txt')).read_text()
rawSimilarity = Path(os.path.join(path, 'Web App\\Data\\similarity.txt')).read_text()

class system(object) : 
    def __init__(self, cusId, rawRating = rawRating, rawSimilarity = rawSimilarity, 
                productListId = productListId.values.tolist(), neighbor = 3) :
        self.ratingData = self.dataProcessing(rawRating)
        self.similarity = self.cleanSimilarity(self.dataProcessing(rawSimilarity))
        self.neighbor = neighbor
        self.productListId = productListId
        self.predictedRating = self.predictRating(cusId, self.similarity, self.ratingData, self.neighbor)
        self.recommend = self.recommendation(self.predictedRating)

    # clean review_score sparse matrix and change its data type to 'list' 
    def dataProcessing(self, data) :
        rawRating = data
        rawRating = rawRating.replace('\n', '')
        rawRating = rawRating.replace('\t', '')
        rawRating = rawRating.replace(' ', '')
        rawRating = rawRating.replace(':', '')
        ratingData = []
        mark = 1
        temp = []
        for i in range(1, len(rawRating)) :
            if rawRating[i] == '(' : 
                temp.append(round(float(rawRating[mark: i]), 2))
                ratingData.append(temp)
                temp = []
                mark = i + 1
            elif rawRating[i] == ',' : 
                temp.append(int(rawRating[mark: i]))
                mark = i + 1
            elif rawRating[i] == ')' : 
                temp.append(int(rawRating[mark: i]))
                mark = i + 1
            if i == len(rawRating) - 1 : 
                temp.append(round(float(rawRating[mark: i]), 2))
                ratingData.append(temp)
        return ratingData
    
    # clean similarity sparse matrix and change its data type to 'list' 
    # remove all items which don't have similarity with other items
    def cleanSimilarity(self, data) : 
        rawSimi = data
        similarity = {}
        for i in range(len(rawSimi)) : 
            if rawSimi[i][0] != rawSimi[i][1] : 
                if rawSimi[i][0] not in similarity :
                    similarity[rawSimi[i][0]] = {rawSimi[i][1] : rawSimi[i][2]}
                else : 
                    similarity[rawSimi[i][0]][rawSimi[i][1]] = rawSimi[i][2]
        return similarity
    
    # check if item rating by input user is available in cleaned similarity matrix 
    # if not : remove it and go on 
    def checkCustomerId(self, similarity, ratedByCus) :
        temp = {}
        for i in ratedByCus : 
            if i in similarity : 
                temp[i] = ratedByCus[i]
        return temp

    # predict rating based on ratings of K items which have the highest similarity with rated items
    # return list items sorted by predicted rating in descending order
    def predictRating(self, cusId, similarity, ratingData, neighbor) :
        ratedByCus = {}
        for i in range(len(ratingData)) : 
            if ratingData[i][1] == cusId : 
                ratedByCus[ratingData[i][0]] = ratingData[i][2]
        ratedByCus = self.checkCustomerId(similarity, ratedByCus) 
        if len(ratedByCus) == 0 : 
            return []
        potentialItem = set()
        for i in ratedByCus : 
            for y in similarity[i] : 
                if y not in ratedByCus : 
                    potentialItem.add(y)
        if len(potentialItem) <= 5 : 
            return potentialItem
        predictedRating = {}
        for i in potentialItem : 
            check = dict(sorted(similarity[i].items(), key=lambda item: item[1], reverse = True))
            neighborAmount = min(neighbor, len(check))
            numerator = 0
            denomerator = 0
            count = 0 
            for y in check : 
                if y in ratedByCus : 
                    numerator += (ratedByCus[y] * check[y])
                denomerator += abs(check[y])
                count += 1
                if count == neighborAmount : 
                    break
            predictedRating[i] = round(numerator / denomerator, 2)
        return dict(sorted(predictedRating.items(), key=lambda item: item[1], reverse = True))

    # decode alternative product id and return its real id
    def decodeProductId(self, alternativeId) : 
        dictionary = self.productListId
        productId = []
        for i in alternativeId : 
            for y in range(len(dictionary)) : 
                if dictionary[y][1] == i : 
                    productId.append(dictionary[y][0])
                    break
        return productId

    # get first 5 item 
    def recommendation(self, predictedRating) : 
        return self.decodeProductId(list(predictedRating)[:5])


# get encoded customer to apply system
def encodeCustomerId(cusId, dictionary = customerListId.values.tolist()) : 
    for i in range(len(dictionary)) : 
        if dictionary[i][0] == cusId : 
            return dictionary[i][1]
    return ''

# get association rules 
def getAssociationRules(product_id, data = prodRules) : 
    data['antecedents'] = data['antecedents'].astype('string')
    data['consequents'] = data['consequents'].astype('string')
    data['antecedents'] = data['antecedents'].str.removeprefix("frozenset({'")
    data['antecedents'] = data['antecedents'].str.removesuffix("'})")
    data['consequents'] = data['consequents'].str.removeprefix("frozenset({'")
    data['consequents'] = data['consequents'].str.removesuffix("'})")
    data = data.values.tolist()
    productAR = []
    for i in range(len(data)) : 
        if data[i][0] == product_id : 
            productAR.append(data[i][1])
    return productAR

# get product's category
def getCategory(productId, dictionary = recommendData) : 
    category = dictionary[dictionary.product_id == productId].product_category_name_english
    return str(category.values[0])

# get customer's segment
def getCustomerSegment(customer_id, data = customerSegment) : 
    return data[data['Customer Id'] == customer_id]['Cluster Name'].values.tolist()[0]

# get product's price
def getPrice(product_id, data = recommendData) : 
    return round(data[data.product_id == product_id].price.values[0], 2)

# get popular product for new customer
def popularProduct(category, recommendData = recommendData, curMonth = dt.datetime.today().month) :
    recommend = recommendData[(recommendData.month == curMonth) & (recommendData.product_category_name_english == category)].sort_values(
                by = ['count', 'review_score'], ascending = [False, False])
    return recommend[['product_id', 'price']][:10].values.tolist()

# get the most expensive product for big spenders
def upSell(category, recommendData = recommendData, curMonth = dt.datetime.today().month) : 
    recommend = recommendData[(recommendData.month == curMonth) & (recommendData.product_category_name_english == category)].sort_values(
                by = ['review_score', 'price'], ascending = [False, False])
    return recommend[['product_id', 'price']][:10].values.tolist()

# get cheapest product which have high review score
def lowPriceProduct(category, recommendData = recommendData, curMonth = dt.datetime.today().month) : 
    recommend = recommendData[(recommendData.month == curMonth) & (recommendData.product_category_name_english == category)].sort_values(
                by = ['price', 'review_score'], ascending = [True, False])
    return recommend[['product_id', 'price']][:10].values.tolist()

app = flask.Flask(__name__, template_folder = os.path.join(path, 'Web App\\Template'))
@app.route('/', methods = ['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
            
    if flask.request.method == 'POST':
        cusId = flask.request.form['customer_id']
        proId = flask.request.form['product_id']
        customer_id = encodeCustomerId(cusId)
        productAR = getAssociationRules(proId)
        category = getCategory(proId)

        recommendFromSystem = system(cusId = customer_id).recommend
        priceFromSystem = []
        for i in recommendFromSystem : 
            priceFromSystem.append(getPrice(i))

        arPrice = []
        for i in productAR : 
            arPrice.append(getPrice(i))
        # get popular product for new customers
        if customer_id == '' : 
            productRecommend = popularProduct(category)
            productRS = []
            priceRS = []
            for i in productRecommend :
                productRS.append(i[0])
                priceRS.append(i[1])
            return flask.render_template('result.html', search_name = 'New Customer', product_idAR = productAR
                                        , priceAR = arPrice, product_idRS = productRS, priceRS = priceRS, cluster_name = 'New Customer')
        else : 
            cusSegment = getCustomerSegment(cusId)
            productRecommend = []
            priceRS = []
            # get popular product and cross sell for Potential 
            if cusSegment == 'Potential' : 
                productRecommend += recommendFromSystem
                priceRS += priceFromSystem
                for i in popularProduct(category)[: 10 - len(recommendFromSystem)] : 
                    productRecommend.append(i[0])
                    priceRS.append(i[1])
            # get low price product for Lost
            elif cusSegment == 'Lost' : 
                productRecommend += recommendFromSystem
                priceRS += priceFromSystem
                for i in lowPriceProduct(category)[: 10 - len(recommendFromSystem)] : 
                    productRecommend.append(i[0])
                    priceRS.append(i[1])
            # cross sell and popular product for Loyal
            elif cusSegment == 'Loyal customers' : 
                productRecommend += recommendFromSystem
                priceRS += priceFromSystem
                for i in popularProduct(category)[: 10 - len(recommendFromSystem)] : 
                    productRecommend.append(i[0])
                    priceRS.append(i[1])
            # up sell for Champions
            elif cusSegment == 'Champions' : 
                for i in upSell(category) : 
                    productRecommend.append(i[0])
                    priceRS.append(i[1])
            return flask.render_template('result.html', search_name = cusId, product_idAR = productAR, priceAR = arPrice
                                        , product_idRS = productRecommend, priceRS = priceRS, cluster_name = cusSegment)

if __name__ == '__main__':
    app.run()