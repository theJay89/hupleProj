from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django import forms
from polls.forms import DocumentForm
from django.shortcuts import redirect 
import os
import sys
import pandas as pd
import numpy as np
from math import sin,cos,asin,acos,radians,atan,pi
import matplotlib.pyplot as plt 
import base64
import io 
from pylab import plot, show, axis, title, xlabel, ylabel, grid, savefig, subplot
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import folium
from IPython.display import display
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold #데이터 분할 / 안써도 무방
import tensorflow as tf
import matplotlib.pyplot as plt


# def home_upload(request):
#     if request.method == 'POST' and request.FILES['myfile']:
#             myfile = request.FILES['myfile']
#             fs = FileSystemStorage()
#             filename = fs.save(myfile.name, myfile)
#             uploaded_file_url = fs.url(filename)
#             return render(request, 'index', {
#                 'uploaded_file_url': uploaded_file_url
#             })
#     return render(request, 'polls/home_upload.html')

def home_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        # <MultiValueDict: {'document': [<InMemoryUploadedFile: Screen Shot 2020-09-28 at 2.05.29 PM.png (image/png)>]}>
        if form.is_valid():
            form.save()
            print(request.FILES['document'])
            # return redirect('index')
            return index(request, request.FILES['document'])
    else:
        form = DocumentForm()
    return render(request, 'polls/home_upload.html', {
        'form': form
    })

def index(request,InMemoryUploadedFile): 
    
    media_path = settings.MEDIA_ROOT
    file_path = os.path.join(media_path, "documents")

    fileName = str(InMemoryUploadedFile)

    print("사용자가 업로드한 파일이름!!: "+ fileName)

    result = calculate(fileName, file_path)

    map = result[0]
    text = result[1]
    graph = result[2]
    name = result[3]
    
    context = {
        'map': map,
        'text': text,
        'graph': graph,
        'name': name,
    }
    return render(request, 'polls/read_write.html', context )



def calculate(fileName, file_path):
        plt.rcParams.update({'figure.max_open_warning': 0})
   
        #%matplotlib inline 
        #지도에 라인그리는 명령어
        #matplotlib로 시각화한 자료를 주피터 노트북 결과창에서 보여줌
    
        #불러올 파일명을 list 저장 추후 웹상에 GPS 데이터를 넣게 되면 그 입력된 파일 명을 가지고 오는 부분으로 대체해야 함. 
    
        file_name = fileName
        # file_name = ['구범모_삼성혈_블록'] #, '구범모_사려니숲길_짚길', '구범모_사려니숲길_1_흙길', '구범모_사려니숲길_2_흙길', 
        #              '구범모_산책길중간_산책로끝_용연다리_블록', '구범모_삼성혈_블록', '구범모_용담공원 산책로_보도', 
        #              '구범모_용연다리에서산책길중간_블록','구범모_우도도항선에서소심한책방_아스', '구범모_종달리_불턱_데크', 
        #              '구범모_종달리_소심한책방_아스', '구범모_종달리불턱_데크', '구범모_종합경기장_1_흙길', '구범모_종합경기장_2_흙길', 
        #              '임성현_내리막_아스', '임성현_사려니숲_짚길', '임성현_사려니숲길_내리막_아스', '임성현_사려니숲길_데크',
        #              '임성현_사려니숲길_흙길', '임성현_소심한책방_불턱_아스', '임성현_용담시민공원_보도', '임성현_용연다리_나무', 
        #              '임성현_용연다리산책길_블록', '임성현_종달리_불턱_데크', '임성현_종달리_불턱_우도항대합실_아스', 
        #              '임성현_종합경기장_1_흙길','임성현_종합경기장_2_흙길']
    
        # name = file_name[0]
        

        print("calculate 메소드로 보내진 파일 이름 : "+ file_name)
        #변수 지정
    
        total_data = []  #최종 계산된 값 저장
        study_data = []  #AI 모델 input 값으로 넣을 값 저장
        di = []          #계산된 거리 저장
        an = []          #계산된 기울기 저장
        al = []          #고도 저장
        la = []          #위도 저장
        lo = []          #경도 저장
        acc_dist = 0     #누적 거리
        tot_point =[]    #지도에서 연결해 줄 점 list 저장
        
        #여러 파일을 순차적으로 불러오기 위해 For 문 사용
        # for name in range(len(file_name)):
            #pandas를 사용해 엑셀 파일의 내용을 data frame의 형식으로 저장함.
            # test_data = pd.read_excel("/Users/hyojinjang/PycharmProjects/wheelC/WH/TestData/계산Data/"+file_name[name] +".xlsx")
        test_data = pd.read_excel(file_path+"/"+file_name)
            #test_list = test_data.iloc[0]
            #gendor ="남"
            
            #파일명에 노면 상태를 넣었기 때문에 파일명에서 뒤에서 두자리를 가지고 와서 각 도로의 형태를 저장
            #엑셀 파일에 노면 상태를 표시하는 값까지 포함되어 입력되면 불필요한 부분임.
        if file_name[-2:] == '아스':
            road_t = '아스팔트'
        else:
            road_t = file_name[-2:]
                
        for i in range(len(test_data)):
                # 데이터의 각 열을 불러와 test_list에 list로 저장함.
            test_list = [j for j in test_data.iloc[i]]
    
            if i==0:
                    #노면 상태를 노면 구름 마찰 계수로 변경하는 if 문
                if road_t == '아스팔트' or road_t == '블록' or road_t == '데크':
                    r_const = 0.01
                elif road_t =='보도' or road_t == '흙길':
                    r_const = 0.02
                else:
                    r_const = 0.05
                    #계산 값을 저장->위도, 경도, 거리, 누적거리, 고도, 경사도, 노면상태, 소요시간, 첫번째 Data는 거리 및 기울기 계산 불가.
                calc_dat = [test_list[0], test_list[1], 0, 0, test_list[2], 0, road_t, test_list[3]]
                pre_lat = test_list[0] # 거리 계산을 위한 이전 지점 위도 값 저장
                pre_lon = test_list[1] # 거리 계산을 위한 이전 지점 경도 값 저장
                pre_alt = test_list[2] # 거리 계산을 위한 이전 지점 고도 값 저장
                total_data.append(calc_dat)
            else:
                if road_t == '아스팔트' or road_t == '블록' or road_t == '데크':
                    r_const = 0.01
                elif road_t =='보도' or road_t == '흙길':
                    r_const = 0.02
                else:
                    r_const = 0.05
                lat = test_list[0]
                lon = test_list[1]
                alt = test_list[2]
                    #time = test_list[3]
                    #하버사인 공식 구현
                dist = acos(cos(radians(90 - pre_lat)) * cos(radians(90 - lat)) + sin(radians(90 - pre_lat)) * sin(radians(90 - lat)) * cos(radians(pre_lon - lon))) * 6731 * 1000
                    #경사도 계산.
                ang = atan((alt - pre_alt) / dist) * 180 / pi
                    #vel = dist / time
                acc_dist = acc_dist + dist
                calc_dat = [test_list[0], test_list[1], dist, acc_dist, test_list[2], ang, road_t, test_list[3]]
                    
                study_dat = [dist, ang, r_const, test_list[3]]
                pre_lat = lat
                pre_lon = lon
                pre_alt = alt
                total_data.append(calc_dat)
                study_data.append(study_dat)
                di.append(acc_dist)
                an.append(ang)
                al.append(alt)
                la.append(lat)
                lo.append(lon)
                    
                la_max = max(la)    
                la_min = min(la)    
                lo_max = max(lo)    
                lo_min = min(lo)
    
        #계산된 누적거리와 고도, 기울기 그래프를 그려줌
            
        plt.figure(figsize=(15,5))    
        plt.subplot(3, 1, 1)    
        plt.plot(di, al)     
        plt.axis(xmin=2, ymin=min(al) - 10)    
        plt.title('Altitude Profile')    
        plt.xlabel('Distance(m)')    
        plt.ylabel('Altitude(m)')    
        plt.grid(True)    
                
        plt.subplot(3, 1, 3)    
        plt.plot(di, an)  # , marker = "o")
            # plt.axis()    
        plt.axis(xmin=1, ymin=min(an) - 3)    
        plt.title('Slope Profile')    
        plt.xlabel('Distance(m)')    
        plt.ylabel('Altitude(°)')    
        plt.grid(True)    

        print("여기가 20번 콜된다고???")
            # plt savefig로 파일을 직접 in-memory에 저장 
        # plt.savefig('chart.jpg', transparent=True)
            
            # pic_IObytes = io.BytesIO()
            # plt.plot(list(range(100)))
            # plt.savefig(pic_IObytes, format='svg')
            # plt.close()
            # plt.savefig(pic_IObytes,  format='png')
            # pic_IObytes.seek(0)
            # graph = base64.b64encode(pic_IObytes.getvalue()).decode("utf-8").replace("\n", "")
            # graph = base64.b64encode(pic_IObytes.getvalue()).decode("utf-8").replace("\n", "")
            # graph = base64.b64encode(pic_IObytes.read())
       
            # plt savefig로 decode한 데이터 html로 넘겨주는 로직 
        svg_file = io.BytesIO()
        plt.savefig(svg_file, format='svg')     # save the file to io.BytesIO
        svg_file.seek(0)
        svg_data = svg_file.getvalue().decode() # retreive the saved data
        graph = svg_data

            
            #GPS에서 받은 위도와 경도 Data를 가지고 지도 위에 표시함.
        m = folium.Map(location=[(la_max + la_min) / 2, (lo_max + lo_min) / 2],
                            zoom_start=16)#, tiles='Stamen Terrain')
        for k in range(len(la)):
        #        if rt[k] == 1:
        #           icon_color = 'gray'
        #       elif rt[k] == 2:
        #           icon_color = 'blue'
        #       elif rt[k] == 3:
        #           icon_color = 'red'
        #       else:
        #           icon_color = 'green'
            folium.CircleMarker([la[k], lo[k]], radius=2,stroke=True, color='red', fill=True,
                                    fill_color='red', line_cap='round').add_to(m)
            point = [la[k], lo[k]]
            tot_point.append(point)
        folium.PolyLine(tot_point, tooltip = 'PolyLine', color='blue').add_to(m)
        mapRendered = m._repr_html_() # 지도 html로 변환하여 pass instead of 'display'
    
        #계산된 값을 dataframe으로 만들고 엑셀로 저장.        
        df = pd.DataFrame(
            total_data,
            columns = ['위도', '경도', '거리', '누적거리', '고도', '경사도', '노면상태', '소요시간'],
        ) 
    
        xlxs_dir = os.path.join("/Users/hyojinjang/PycharmProjects/wheelC/WH/TestData/최종Data/" + file_name+"_연습.xlsx")
        df.to_excel(xlxs_dir)
                
        total_data.clear()
    
            
        #모델에서 계산할 입력Data를 Dataframe으로 만든다.    
        df_study = pd.DataFrame(
            study_data,
            columns = ['거리','경사도', '노면구름계수', '소요시간'],
        ) 
    
        # xlxs_dir = os.path.join("C:/Huple_업무/SW 지원 사업/휠체어 이동시간/휠체어 Test data/학습Data/휠체어_학습Data.xlsx")
        # df.to_excel(xlxs_dir)
    
        #입력 Data를 저장
        dataset = df_study.values
        X = dataset[:,0:3]
        #Y = dataset[:,3]
    
        #접근 가능 여부 판단
        unable_dist = 0
        X_fin = []
        X_rem = []
        for n in range(len(X)):
            if X[n][1] > 7 or X[n][1] < -7:
                unable_dist = unable_dist + X[n][0]
                X_rem.append(X[n])
            else:
                data_fin = [X[n][0],X[n][1],X[n][2]]
                X_fin.append(data_fin)
    
        #모델 로딩
        model = load_model('/Users/hyojinjang/PycharmProjects/wheelC/WH/TestData/model/wheel_chair_model.h5')
    
        #접근 가능 부분 계산 진행
        Y_prediction = model.predict(X_fin).flatten()    
        Y_pre_tot = np.sum(Y_prediction)
        #Y_tot = sum(Y)
        
        resultText = "총 거리 {:.1f} KM, 예상 이동 소요 시간은 {:.1f} 분이며 휠체어 이동이 어려운 구간은 총 {:.1f} KM 입니다.".format(acc_dist/1000, Y_pre_tot/60, unable_dist/1000)
        #print("총 거리 {:.1f} KM, 예상 이동 소요 시간은 {:.1f} 분이며 휠체어 이동이 어려운 구간은 총 {:.1f} KM 입니다.".format(acc_dist/1000, Y_pre_tot/60, unable_dist/1000))

        plt.close('all')
        
        return mapRendered, resultText, graph, file_name

# calculate()