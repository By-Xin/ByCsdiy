# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import urllib.request
import json
import zlib
import xml.dom.minidom

def input_url():
    print("input bilibili url:")
    url = input()
    return url

def get_bv_from_url(url):
    text = url.split("/")
    urltext=filter(lambda x: x.startswith("BV"),text)
    return urltext

def get_cid_from_bv(bv):
    api = str("https://api.bilibili.com/x/web-interface/view?bvid=")
    url = str(api) + str(bv)
    response = urllib.request.urlopen(url) #response 是发送请求后的返回内容
    string = response.read() #string是对response的解读
    html = string.decode("utf-8") #需要对string进行编码变成自然语言
    doc = json.loads(html)
    return doc["data"]["cid"] #读取到其中的cid

def get_xml_from_cid(cid):
    api = "https://api.bilibili.com/x/v1/dm/list.so?oid="
    url = api + str(cid)
    response = urllib.request.urlopen(url)
    response.getheader("Content-Encoding")
    string = response.read()
    html = zlib.decompress(string, wbits=-zlib.MAX_WBITS).decode('utf-8')
    xml2 = xml.dom.minidom.parseString(html)
    xml_pretty_str = xml2.toprettyxml()
    return xml_pretty_str

def filter_line(line):
    if line.startswith("\t<d"):
        return True
    else :
        return False

def clean_line(line):
    start = line.find('">')
    end = line.find("</d>")
    puretext=line[start+2:end]+"\n"
    return puretext

def get_bar_from_xml(doc):
    lines = doc.split("\n")
    it = filter(filter_line, lines)
    it = map(clean_line, it)
    return list(it)

def output(bar):
    f=open(r"barrage.txt","w")
    f.writelines(bar)
    f.close()
    print("Success! :-) \nFind report as ***barrage.txt*** at root dir")
    return

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    bili = input_url()
    bv = list(get_bv_from_url(bili))[0][:12]
    cid = str(get_cid_from_bv(bv))
    xml = get_xml_from_cid(cid)
    bar = get_bar_from_xml(xml)
    output(bar)



#https://www.bilibili.com/video/BV1nM4y1f7s4/?spm_id_from=333.1007.tianma.1-1-1.click&vd_source=8a00dab0be94d29388f2286892ba8d50

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
