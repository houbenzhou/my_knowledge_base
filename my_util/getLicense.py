import random
import re
from http import cookiejar
from urllib import parse
from urllib import request

string_seed = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

email_http = "http://24mail.chacuo.net/"
cookie = cookiejar.CookieJar()


# 创建cookie
def get_email_opener():
    handler = request.HTTPCookieProcessor(cookie)
    return request.build_opener(handler)


def get_email_address():
    #req = request.Request(email_http)
    #text = tempCookie.open(req).read().decode("utf-8")
    #email_prefix = re.findall('<input id="converts" name="converts" type="text" value="([a-zA-Z0-9]+)"', text)[0]
    #email_name = '%s%s' % (email_prefix, '@chacuo.net')
	sa = []
	for i in range(10):
		sa.append(random.choice(string_seed))
	email_name = ''.join(sa)+'@chacuo.net'
	return email_name


# 获取邮箱
tempCookie = get_email_opener()
email = get_email_address()
print('邮箱地址:' + email)  # 输出邮箱地址

# 注册账号
arcgisHeaders = {
    'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'Referer': r'https://www.arcgis.com/features/free-trial-form.html',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Host': 'webappsproxy.esri.com',
    'Origin': 'https://www.arcgis.com',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    # 'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive'
}
arcgisUrl = r'https://webappsproxy.esri.com/passthrough?https://billing.arcgis.com/sms/rest/activation/create'
signData = {
    'firstName': 'a',
    'lastName': 'a',
    'email': email,
    'confirm_email': email,
    'zipCode': '550560',
    'phoneNumber': '110',
    'company': 'Aaxico',
    'industry': 'AEC',
    'organizationalRole': 'Student',
    'undefined': 'Student',
    'demandbase_company_name': 'Aaxico China',
    'demandbase_marketing_alias': 'Aaxico',
    'demandbase_city': 'Guangzhou',
    'demandbase_country_name': 'China',
    'demandbase_state': '91',
    'demandbase_zip': '',
    'demandbase_country': 'CN',
    'demandbase_street_address': 'Low',
    'demandbase_traffic': 'Rm 706b Lasony Business Center',
    'demandbase_phone': '+86 20 8733 7907',
    'demandbase_primary_naics': '',
    'demandbase_demandbase_sid': '115056627',
    'demandbase_primary_sic': '3728',
    'demandbase_industry': 'Aerospace & Defense',
    'demandbase_sub_industry': 'Aircraft',
    'demandbase_employee_count': '5',
    'demandbase_employee_range': 'Small',
    'demandbase_annual_sales': '5500000',
    'demandbase_revenue_range': '$5M - $10M',
    'demandbase_b2b': 'true',
    'demandbase_fortune_1000': 'false',
    'demandbase_forbes_2000': 'false',
    'demandbase_latitude': '23.1167',
    'demandbase_longitude': '113.25',
    'demandbase_stock_ticker': '',
    'demandbase_web_site': '',
    'reason': 'ArcGIS free trial',
    'product': 'ArcGIS Desktop',
    'origin': '',
    'f': 'json'
}
#
data = parse.urlencode(signData).encode('utf-8')
arcgireq = request.Request(arcgisUrl, headers=arcgisHeaders, data=data)
url = tempCookie.open(arcgireq)
signResult = url.read().decode("utf-8")
print('邮箱注册结果:' + signResult)  # 输出注册结果
signCode = re.findall('"code":([0-9]+),', signResult)[0]
if int(signCode) != 200:
    print("邮箱已存在，重试吧")
    exit(0)

activationcode = re.findall('"message":"(.*)","status', signResult)[0]

# 获取cookie
getCookieHeader = {
    'authority': 'www.arcgis.com',
    'method': 'GET',
    'path': '/sharing/geoip.jsp?f=json',
    'scheme': 'https',
    'accept': '*/*',
    'Connection': 'keep-alive',
    'accept-language': 'zh-CN,zh;q=0.9',
    'content-type': 'application/x-www-form-urlencoded',
    'referer': 'http://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
    'user-agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
}
#tempCookie.open(request.Request("https://www.arcgis.com/sharing/geoip.jsp?f=json",
 #                               data=parse.urlencode({'f': 'json'}).encode('utf-8'), headers=getCookieHeader)).read()

#arcgisCookieString = re.findall("(JSESSIONID=[a-zA-Z0-9]+)\sfor", str(cookie))[0]


# 获取可用用户名
def getusername():
    for i in range(100):
        sa = []
        for i in range(50):
            sa.append(random.choice(string_seed))
        user_name = ''.join(sa)
        get_name_url = 'http://www.arcgis.com/sharing/proxy?https://billing.arcgis.com/sms/rest/activation/checkUsername?username=%s&f=json&agolOnly' % user_name
        get_user_header = {
            'Host': 'www.arcgis.com',
            'Connection': 'keep-alive',
            # 'Accept-Encoding': 'gzip, deflate',
            # 'path': r'/sharing/proxy?https://billing.arcgis.com/sms/rest/activation/checkUsername?username=%s&f=json&agolOnly' % user_name,
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Accept': r'application/json, text/javascript, */*; q=0.01',
            'user-agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'Referer': r'http://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
            # 'Cookie': arcgisCookieString,
            'X-Requested-With': 'XMLHttpRequest'
        }
        user_name_post_datas = {
            'f': 'json',
            r'https://billing.arcgis.com/sms/rest/activation/checkUsername?username': user_name
        }
        temp_datas = parse.urlencode(user_name_post_datas) + '&agolOnly'
        user_name_post_data = temp_datas.encode('utf-8')
        getUserResult = request.Request(get_name_url, headers=get_user_header, data=user_name_post_data, method="GET")
        res = tempCookie.open(getUserResult)
        user_name_result = res.read().decode("utf-8")
        is_available = re.findall('"available":([a-zA-Z]+)}', user_name_result)[0]
        if bool(is_available):
            return user_name
    print("你这也太背了，100次用户名都拿不到一个可用的，重来吧！")


userName = getusername()
password = 'Aa12345621343258hskdjjfg'
print('用户名: ' + userName)  # 输出用户名

url = 'https://www.arcgis.com/sharing/rest/oauth2/signout'
headers = {
    'authority': 'www.arcgis.com',
    'path': r'/sharing/rest/oauth2/signout',
    'scheme': 'https',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    # 'cookie': arcgisCookieString,
    'referer': 'https://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest'
}
tempCookie.open(request.Request(url, headers=headers, method='GET')).read().decode('utf-8')

# 注册账号
sign_url = r'http://www.arcgis.com/sharing/proxy?https://billing.arcgis.com/sms/rest/activation'
sign_headers = {
    'Accept': r'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Content-Type': r'application/x-www-form-urlencoded; charset=UTF-8',
    # 'Cookie': arcgisCookieString,
    'Host': 'www.arcgis.com',
    'Origin': r'http://www.arcgis.com',
    'Referer': r'http://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
    'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
sign_datas = {
    'username': userName,
    'password': password,
    'confirmationPassword': password,
    'email': email,
    'organization': 'Aaxico',
    'identifyQuestion': '您最喜欢的书的书名是什么?',
    'identifyQuestionID': '12',
    'identifyAnswer': 'aaaaaa',
    'legalTerms': 'on',
    'firstName': 'a',
    'lastName': 'a',
    'phoneNumber': '-',
    'languageCode': 'en',
    'format': 'json',
    'activationToken': activationcode
}
sign_data = parse.urlencode(sign_datas).encode("utf-8")
getSignResult = request.Request(sign_url, headers=sign_headers, data=sign_data)
signReturn = tempCookie.open(getSignResult).read().decode("utf-8")
print('注册第一步结果: ' + signReturn)
sign2_url = 'http://www.arcgis.com/sharing/proxy?https://www.arcgis.com/sharing/rest/community/signup'
sign2_headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    # 'Cookie': arcgisCookieString,
    'Host': 'www.arcgis.com',
    'Origin': 'http://www.arcgis.com',
    'Referer': 'http://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
sign2_datas = {
    'username': userName,
    'password': password,
    'email': email,
    'fullname': 'a a',
    'firstName': 'a',
    'lastName': 'a',
    'securityQuestionIdx': '12',
    'securityAnswer': 'aaaaaa',
    'activationCode': activationcode,
    'referer': 'arcgis.com',
    'provider': 'arcgis',
    'usertype': 'both',
    'f': 'json'
}
sign2_data = parse.urlencode(sign2_datas).encode('utf-8')
sign2_req = request.Request(sign2_url, data=sign2_data, headers=sign2_headers)
sign2_open_result = tempCookie.open(sign2_req)
sign2_read_result = sign2_open_result.read()
sing2_result = sign2_read_result.decode('utf-8')
print('注册第二步结果: ' + sing2_result)
token = re.findall('"token":"(.*)","', sing2_result)[0]

sign3_url = 'https://www.arcgis.com/sharing/proxy?https://www.arcgis.com/sharing/rest/portals/activate?token=%s&code=%s&f=json' % (
    token, activationcode)
sign3_header = {
    'authority': 'www.arcgis.com',
    'path': r'/sharing/proxy?https://www.arcgis.com/sharing/rest/portals/activate?token=%s&code=%s&f=json' % (
        token, activationcode),
    'scheme': 'https',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-language': 'zh-CN,zh;q=0.9',
    # 'cookie': arcgisCookieString,
    'referer': 'https://www.arcgis.com/features/login/activation.html?activationcode=446f8eea39118c0fa86d',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest'
}
sign3_datas = {
    'https://www.arcgis.com/sharing/rest/portals/activate?token': token,
    'code': activationcode,
    'f': 'json'
}
sign3_result = tempCookie.open(
    request.Request(sign3_url, headers=sign3_header, data=parse.urlencode(sign3_datas).encode('utf-8'),
                    method='GET')).read().decode('utf-8')
print("注册第三步结果：" + sign3_result)

url = r'https://www.arcgis.com/sharing/proxy?https://www.arcgis.com/sharing/rest/portals/self/'
headers = {
    'authority': 'www.arcgis.com',
    'path': '/sharing/proxy?https://www.arcgis.com/sharing/rest/portals/self/',
    'scheme': 'www.https.com',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-language': 'zh-CN,zh;q=0.9',
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
    # 'cookie':arcgisCookieString,
    'origin': 'https://www.arcgis.com',
    'referer': 'https://www.arcgis.com/features/login/activation.html?activationcode=%s' % activationcode,
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest'
}
datas = {
    '?https://go.pardot.com/l/82202/2015-10-09/c3dgt': '',
    'company': 'Aaxico',
    'email': email
}
reqResult = tempCookie.open(
    request.Request(url, data=parse.urlencode(datas).encode('utf-8'), headers=headers)).read().decode('utf-8')
# 登陆
oa_url = 'https://www.arcgis.com/sharing/rest/oauth2/authorize?client_id=arcgisonline&redirect_uri=https://m.arcgis.com/home/postsignin.html&showSocialLogins=true&hideEnterpriseLogins=false&response_type=token&display=iframe&parent=https://m.arcgis.com&expiration=20160&locale=zh-cn'
oa_headers = {
    'authority': 'www.arcgis.com',
    'method': 'get',
    'path': '/sharing/rest/oauth2/authorize?client_id=arcgisonline&redirect_uri=https://m.arcgis.com/home/postsignin.html&showSocialLogins=true&hideEnterpriseLogins=false&response_type=token&display=iframe&parent=https://m.arcgis.com&expiration=20160&locale=zh-cn',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cache-control': 'max-age=0',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
}
oa_datas = {
    'client_id': 'arcgisonline',
    'redirect_uri': 'https://www.arcgis.com/home/postsignin.html',
    'showSocialLogins': 'true',
    'hideEnterpriseLogins': 'false',
    'response_type': 'token',
    'display': 'iframe',
    'parent': 'https://www.arcgis.com',
    'locale': 'zh-cn',
}
oa_data = parse.urlencode(oa_datas).encode('utf-8')
oa_result = request.urlopen(request.Request(oa_url, data=oa_data, headers=oa_headers)).read().decode('utf-8')
token = re.findall('"oauth_state":"(.*)","client_id"', oa_result)[0]

login_url = 'https://www.arcgis.com/sharing/oauth2/signin'
login_headers = {
    'authority': 'www.arcgis.com',
    'method': 'POST',
    'path': r'/sharing/oauth2/signin',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cache-control': 'max-age=0',
    'content-type': 'application/x-www-form-urlencoded',
    # 'cookie': arcgisCookieString,
    'origin': 'https://www.arcgis.com',
    'referer': 'https://www.arcgis.com/sharing/rest/oauth2/authorize?client_id=arcgisonline&redirect_uri=https://www.arcgis.com/home/postsignin.html&showSocialLogins=true&hideEnterpriseLogins=false&response_type=token&display=iframe&parent=https://www.arcgis.com&expiration=20160&locale=zh-cn',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
}
login_datas = {
    'user_orgkey': '',
    'username': userName,
    'password': password,
    'oauth_state': token,
    'persist': 'true'
}
login_data = parse.urlencode(login_datas).encode('utf-8')
login_req = request.Request(login_url, headers=login_headers, data=login_data)
login_result = tempCookie.open(login_req).read().decode('utf-8')

arcgisCookieString = re.findall("(esri_auth=.*)\sfor\s.arcgis.com", str(cookie))[0]
uncodeString = parse.unquote(arcgisCookieString)
signToken = re.findall('"token":"(.*)","culture', uncodeString)[0]
id = re.findall(',"id":"(.*)"}', uncodeString)[0]


# 获取可用组织名称
def get_org_name():
    for i in range(100):
        sa = []
        for j in range(15):
            sa.append(random.choice(string_seed))
        temp_org_name = ''.join(sa)
        org_name_url = 'https://www.arcgis.com/sharing/rest/portals/isUrlKeyAvailable?urlKey=%s&f=json&token=%s' % (
            temp_org_name, signToken)
        org_headers = {
            'authority': 'www.arcgis.com',
            'path': '/sharing/rest/portals/isUrlKeyAvailable?urlKey=%s&f=json&token=' % temp_org_name,
            'scheme': 'https',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'referer': 'https://www.arcgis.com/home/setup.html',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            # 'cookie': arcgisCookieString
        }
        org_datas = {
            'urlKey': temp_org_name,
            'f': 'json',
            'token': signToken
        }
        org_data = parse.urlencode(org_datas).encode('utf-8')
        org_req = request.Request(org_name_url, headers=org_headers, data=org_data, method='GET')
        org_result = tempCookie.open(org_req).read().decode('utf-8')
        print("获取可用组织名结果：" + org_result)
        is_org_available = re.findall('"available":(.*),"', org_result)[0]
        if bool(is_org_available):
            temp_org_name = re.findall('"urlKey":"(.*)"}', org_result)[0]
            return temp_org_name
    print("获取组织名失败，请重试。")
    exit(0)


# 设置组织
org_name = get_org_name()
set_org_url = 'https://www.arcgis.com/sharing/rest/portals/self/update'
set_org_header = {
    'authority': 'www.arcgis.com',
    'path': '/sharing/rest/portals/self/update',
    'scheme': 'https',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'content-type': 'application/x-www-form-urlencoded',
    'origin': 'https://www.arcgis.com',
    'referer': 'https://www.arcgis.com/home/setup.html',
    # 'cookie': arcgisCookieString,
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
}
set_org_datas = {
    'id': id,
    'name': 'Aaxico',
    'access': 'private',
    'allSSL': 'false',
    'culture': '',
    'canSharePublic': 'true',
    'canSearchPublic': 'true',
    'defaultBasemap': '',
    'defaultExtent': '',
    'featuredGroups': 'null',
    'homePageFeaturedContentCount': '12',
    'rotatorPanels': 'null',
    'analysisLayersGroupQuery': 'title:"Living Atlas Analysis Layers" AND owner:esri',
    'galleryTemplatesGroupQuery': 'title:"Gallery Templates" AND owner:esri_zh',
    'stylesGroupQuery': 'title:"Esri Styles" AND owner:esri_zh',
    'urlKey': org_name,
    'commentsEnabled': 'true',
    'geocodeService': r'[{"url":"https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer","northLat":"Ymax","southLat":"Ymin","eastLon":"Xmax","westLon":"Xmin","name":"Esri World Geocoder","batch":true,"placefinding":true,"suggest":true}]',
    'printServiceTask': r'{"url":"https://utility.arcgisonline.com/arcgis/rest/services/Utilities/PrintingTools/GPServer/Export%20Web%20Map%20Task"}',
    'geometryService': r'{"url":"https://utility.arcgisonline.com/arcgis/rest/services/Geometry/GeometryServer"}',
    'routeServiceLayer': r'{"url":"https://route.arcgis.com/arcgis/rest/services/World/Route/NAServer/Route_World","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'canSignInIDP': 'true',
    'canSignInArcGIS': 'true',
    'useStandardizedQuery': 'true',
    'portalProperties': '{"mustWelcome":true,"links":{}}',
    'mfaEnabled': 'false',
    'mfaAdmins': '[]',
    'metadataEditable': 'false',
    'metadataFormats': '',
    'updateUserProfileDisabled': 'false',
    'useVectorBasemaps': 'false',
    'eueiEnabled': 'false',
    'clearEmptyFields': 'true',
    'asyncRouteService': r'{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/Route/GPServer","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'closestFacilityService': r'{"url":"https://route.arcgis.com/arcgis/rest/services/World/ClosestFacility/NAServer/ClosestFacility_World","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'asyncClosestFacilityService': r'{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/ClosestFacility/GPServer/FindClosestFacilities","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'serviceAreaService': '{"url":"https://route.arcgis.com/arcgis/rest/services/World/ServiceAreas/NAServer/ServiceArea_World","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'asyncServiceAreaService': r'{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/ServiceAreas/GPServer/GenerateServiceAreas","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'syncVRPService': r'{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/VehicleRoutingProblemSync/GPServer/EditVehicleRoutingProblem","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'asyncVRPService': '{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/VehicleRoutingProblem/GPServer/SolveVehicleRoutingProblem","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'asyncLocationAllocationService': '{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/LocationAllocation/GPServer","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'routingUtilitiesService': '{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/Utilities/GPServer"}',
    'trafficService': '{"url":"https://traffic.arcgis.com/arcgis/rest/services/World/Traffic/MapServer"}',
    'asyncODCostMatrixService': '{"url":"https://logistics.arcgis.com/arcgis/rest/services/World/OriginDestinationCostMatrix/GPServer","defaultTravelMode":"FEgifRtFndKNcJMJ"}',
    'contacts': '["%s"]' % userName,
    'f': 'json',
    'token': signToken
}
org_result = tempCookie.open(request.Request(set_org_url, headers=set_org_header,
                                             data=parse.urlencode(set_org_datas).encode('utf-8'))).read().decode(
    'utf-8')
print('设置组织结果:' + org_result)

# 跳转组织已删除
license_result = 'error'


def reTryGetLicense():
    global cookie, arcgisCookieString, uncodeString, signToken, id, license_url, license_headers, license_Data, license_result
    cookie = cookiejar.CookieJar()
    cookieHandle = request.build_opener(request.HTTPCookieProcessor(cookie))
    orgPage = cookieHandle.open(request.Request(
        'https://%s.maps.arcgis.com/sharing/rest/oauth2/authorize?client_id=arcgisonline&redirect_uri=https://%s.maps.arcgis.com/home/postsignin.html&showSocialLogins=true&hideEnterpriseLogins=false&response_type=token&display=iframe&parent=https://%s.maps.arcgis.com&expiration=20160&locale=zh-cn' % (
            org_name, org_name, org_name))).read().decode('utf-8')
    oauth_state = re.findall('"oauth_state":"(.*)","client_id"', orgPage)[0]
    orgSignInUrl = 'https://%s.maps.arcgis.com/sharing/oauth2/signin' % org_name
    orgSignInHeader = {
        'authority': '%s.maps.arcgis.com' % org_name,
        'path': '/sharing/oauth2/signin',
        'scheme': 'https',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'max-age=0',
        'origin': 'https://4wkb3u7mpqjltlo.maps.arcgis.com',
        'referer': 'https://%s.maps.arcgis.com/sharing/rest/oauth2/authorize?client_id=arcgisonline&redirect_uri=https://%s.maps.arcgis.com/home/postsignin.html&showSocialLogins=true&hideEnterpriseLogins=false&response_type=token&display=iframe&parent=https://%s.maps.arcgis.com&expiration=20160&locale=zh-cn' % (
            org_name, org_name, org_name),
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
    }
    orgSignInDatas = {
        'user_orgkey': '',
        'username': userName,
        'password': password,
        'oauth_state': oauth_state
    }
    signInData = parse.urlencode(orgSignInDatas).encode('utf-8')
    signInResult = cookieHandle.open(
        request.Request(orgSignInUrl, headers=orgSignInHeader, data=signInData)).read().decode(
        'utf-8')
    arcgisCookieString = re.findall("(esri_auth=.*)\sfor\s.arcgis.com", str(cookie))[0]
    uncodeString = parse.unquote(arcgisCookieString)
    signToken = re.findall('"token":"(.*)","culture', uncodeString)[0]
    id = re.findall(',"id":"(.*)"}', uncodeString)[0]
    license_url = r'https://%s.maps.arcgis.com/sharing/rest/content/listings/2d2a9c99bb2a43548c31cd8e32217af6/provisionUserEntitlements' % org_name
    license_headers = {
        'authority': '%s.maps.arcgis.com' % org_name,
        'path': '/sharing/rest/content/listings/2d2a9c99bb2a43548c31cd8e32217af6/provisionUserEntitlements',
        'scheme': 'https',
        'accept': '*/*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'content-type': 'application/x-www-form-urlencoded',
        'origin': 'https://%s.maps.arcgis.com' % org_name,
        'referer': 'https://%s.maps.arcgis.com/home/organization.html' % org_name,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    }
    # "desktopAdvN","spatialAnalystN","3DAnalystN","networkAnalystN","geostatAnalystN","dataReviewerN","workflowMgrN","dataInteropN","publisherN","imageAnalystN"
    license_Data = {
        'userEntitlements': r'{"users":["%s"],"entitlements":["desktopAdvN","spatialAnalystN","3DAnalystN","networkAnalystN","geostatAnalystN","dataReviewerN","workflowMgrN","dataInteropN","publisherN","imageAnalystN"]}' % userName,
        'suppressCustomerEmail': 'true',
        'f': 'json',
        'token': signToken
    }
    license_result = cookieHandle.open(request.Request(license_url, headers=license_headers,
                                                       data=parse.urlencode(license_Data).encode(
                                                           'utf-8'))).read().decode(
        'utf-8')
    print('分配许可结果:' + license_result, end='')
    return license_result


while str(license_result).find('error') != -1 :
    if(license_result!='error'):
        print('分配失败，重试中，请耐心等待, 一般失败5次后才会成功。')
    license_result = reTryGetLicense()
print('分配许可成功')
print('----------------------')
print('用户名: ' + userName)
print('密码: ' + password)
