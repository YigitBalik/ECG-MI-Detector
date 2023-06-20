import os
import tempfile
from PyPDF2 import PdfWriter, PdfReader
import numpy as np
from svgpathtools import svg2paths2
import cmath
import scipy as sp
import pandas as pd
import re

class SignalExtractor(object):

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def pol2cart(self,polar):
        rho = polar[0]
        phi = polar[1]
        x = rho * np.cos(phi)
        y= rho * np.sin(phi)
        return (x,y)
    
    def pars_info(self, file):
        paths, attributes, svg_attributes = svg2paths2(file)
        pth_tmp = paths[411:]
        return pth_tmp

    def chang_corrd(self, pth_tmp):
        lst_tmp = []
        n = 0
        for i in pth_tmp:
            path_tmp = [pth_tmp[n][0::2]]
            lst_tmp = lst_tmp + path_tmp
            n += 1
            path_tmp = []
        return lst_tmp

    def corrd2data(self, lst_tmp):
        x_list = []
        y_list = []
        n = 0
        for i in lst_tmp:
            x = []
            y = []
            for i[n] in i:
                real_num_s = i[n][0].real
                imag_num_s = i[n][0].imag

                real_num_e = i[n][1].real
                imag_num_e = i[n][1].imag

                start_cn = complex(real_num_s, imag_num_s)
                end_cn = complex(real_num_e, imag_num_e)

                start_pol = cmath.polar(start_cn)
                end_pol = cmath.polar(end_cn)

                
                start_poi = self.pol2cart(start_pol)
                end_poi = self.pol2cart(end_pol)
            
                start_poi_x = [start_poi[0]]
                end_poi_x = [end_poi[0]]
                start_poi_y = [start_poi[1]]
                end_poi_y = [end_poi[1]]

                x = x + start_poi_x + end_poi_x
                start_poi_x = []
                end_poi_x = []

                y = y + start_poi_y + end_poi_y
                start_poi_y = []
                end_poi_y = []

            x_list.append(x[2:])
            y_list.append(y[2:])
        return x_list, y_list
    
    def inter(self, x_list, y_list):
        y_array = np.array(y_list)
        x_array = np.array(x_list)
        
        new_len = 1501
        new_x =  np.linspace(x_array.min(), x_array.max(),new_len)
        new_y = sp.interpolate.interp1d(x_array,y_array)(new_x)
        x_list = new_x.tolist()
        y_list = new_y.tolist()
        return x_list, y_list

    def inter2(self, x_list, y_list):
        y_array = np.array(y_list)
        x_array = np.array(x_list)

        new_len = 5000

        new_x = np.linspace(x_array.min(), x_array.max(),new_len)
        new_y = sp.interpolate.interp1d(x_array,y_array)(new_x)
        x_list = new_x.tolist()
        y_list = new_y.tolist()

        return x_list, y_list
    
    def fix_data(self, x_list, y_list):
        bl1 = y_list[0][-1]
        bl2 = y_list[1][-1]
        bl3 = y_list[2][-1]
        bl4 = y_list[3][-1]
        bl5 = y_list[4][-1]
        bl6 = y_list[5][-1]
        

        dis_fn6 = bl1 - y_list[6][0]

        if dis_fn6 > 0:
            y_list[6] = list(map(lambda y:y + dis_fn6, y_list[6]))
            x_list[6], y_list[6] = self.inter2(x_list[6],y_list[6])
            y_list[6] = list(map(lambda y: y - y_list[6][0], y_list[6]))
        else:
            y_list[6] = list(map(lambda y: y - dis_fn6, y_list[6]))
            x_list[6], y_list[6] = self.inter2(x_list[6],y_list[6])
            y_list[6] = list(map(lambda y: y - y_list[6][0], y_list[6]))

        dis_fn7 = bl2 - y_list[7][0]
        if dis_fn7 > 0:
            y_list[7] = list(map(lambda y: y + dis_fn7, y_list[7]))
            x_list[7], y_list[7] = self.inter2(x_list[7],y_list[7])
            y_list[7] = list(map(lambda y: y - y_list[7][0], y_list[7]))
        else:
            y_list[7] = list(map(lambda y: y - dis_fn7, y_list[7]))
            x_list[7], y_list[7] = self.inter2(x_list[7],y_list[7])
            y_list[7] = list(map(lambda y: y - y_list[7][0], y_list[7]))

        dis_fn8 = bl3 - y_list[8][0]
        if dis_fn8 > 0:
            y_list[8] = list(map(lambda y: y + dis_fn8, y_list[8]))
            x_list[8], y_list[8] = self.inter2(x_list[8],y_list[8])
            y_list[8] = list(map(lambda y: y - y_list[8][0], y_list[8]))
        else:
            y_list[8] = list(map(lambda y: y - dis_fn8, y_list[8]))
            x_list[8], y_list[8] = self.inter2(x_list[8],y_list[8])
            y_list[8] = list(map(lambda y: y - y_list[8][0], y_list[8]))

        dis_fn9 = bl4 - y_list[9][0]
        if dis_fn9 > 0:
            y_list[9] = list(map(lambda y: y + dis_fn9, y_list[9]))
            x_list[9], y_list[9] = self.inter2(x_list[9],y_list[9])
            y_list[9] = list(map(lambda y: y - y_list[9][0], y_list[9]))
        else:
            y_list[9] = list(map(lambda y: y - dis_fn9, y_list[9]))
            x_list[9], y_list[9] = self.inter2(x_list[9],y_list[9])
            y_list[9] = list(map(lambda y: y - y_list[9][0], y_list[9]))

        dis_fn10 = bl5 - y_list[10][0]
        if dis_fn10 > 0:
            y_list[10] = list(map(lambda y: y + dis_fn10, y_list[10]))
            x_list[10], y_list[10] = self.inter2(x_list[10],y_list[10])
            y_list[10] = list(map(lambda y: y - y_list[10][0], y_list[10]))
        else:
            y_list[10] = list(map(lambda y: y - dis_fn10, y_list[10]))
            x_list[10], y_list[10] = self.inter2(x_list[10],y_list[10])
            y_list[10] = list(map(lambda y: y - y_list[10][0], y_list[10]))

        dis_fn11 = bl6 - y_list[11][0]
        if dis_fn11 > 0:
            y_list[11] = list(map(lambda y: y + dis_fn11, y_list[11]))
            x_list[11], y_list[11] = self.inter2(x_list[11],y_list[11])
            y_list[11] = list(map(lambda y: y - y_list[11][0], y_list[11]))
        else:
            y_list[11] = list(map(lambda y: y - dis_fn11, y_list[11]))
            x_list[11], y_list[11] = self.inter2(x_list[11],y_list[11])
            y_list[11] = list(map(lambda y: y - y_list[11][0], y_list[11]))

        return x_list, y_list
    
    def adj(self, x_list,y_list):
        candid_1 = y_list[0]
        max_value_1 = max(candid_1)
        min_value_1 = min(candid_1)
        base_1 = abs(max_value_1 - min_value_1)
        y_list[6] = [x/base_1 for x in y_list[6]]
        
        candid_2 = y_list[1]
        max_value_2 = max(candid_2)
        min_value_2 = min(candid_2)
        base_2 = abs(max_value_2 - min_value_2)
        y_list[7] = [x/base_2 for x in y_list[7]]
        
        candid_3 = y_list[2]
        max_value_3 = max(candid_3)
        min_value_3 = min(candid_3)
        base_3 = abs(max_value_3 - min_value_3)
        y_list[8] = [x/base_3 for x in y_list[8]]

        candid_4 = y_list[3]
        max_value_4 = max(candid_4)
        min_value_4 = min(candid_4)
        base_4 = abs(max_value_4 - min_value_4)
        y_list[9] = [x/base_4 for x in y_list[9]]

        candid_5 = y_list[4]
        max_value_5 = max(candid_5)
        min_value_5 = min(candid_5)
        base_5 = abs(max_value_5 - min_value_5)
        y_list[10] = [x/base_5 for x in y_list[10]]

        candid_6 = y_list[5]
        max_value_6 = max(candid_6)
        min_value_6 = min(candid_6)
        base_6 = abs(max_value_6 - min_value_6)
        y_list[11] = [x/base_6 for x in y_list[11]]
        
        return x_list, y_list
    
    def mk_pECG(self, x_list, y_list):
        p_df_6 = pd.DataFrame(y_list[6])
        p_df_7 = pd.DataFrame(y_list[7])
        p_df_8 = pd.DataFrame(y_list[8])
        p_df_9 = pd.DataFrame(y_list[9])
        p_df_10 = pd.DataFrame(y_list[10])
        p_df_11 = pd.DataFrame(y_list[11])

        return p_df_6, p_df_7, p_df_8, p_df_9, p_df_10, p_df_11
    
    def execute(self, file, file_name):
        pth_tmp = self.pars_info(file)
        lst_tmp = self.chang_corrd(pth_tmp)
        x_list, y_list = self.corrd2data(lst_tmp)
        # print(len(x_list[6]), len(y_list[6]))
        x_list, y_list = self.fix_data(x_list, y_list)
        # print(len(x_list[6]), len(y_list[6]))
        x_list, y_list = self.adj(x_list, y_list)
        # print(len(x_list[6]), len(y_list[6]))

        
        p_df_6,p_df_7,p_df_8,p_df_9, p_df_10,p_df_11 = self.mk_pECG(x_list,y_list)

        signals_half = np.zeros((5000,6))
        signals_half[:, 0] = p_df_6.values.reshape(-1)
        signals_half[:, 1] = p_df_7.values.reshape(-1)
        signals_half[:, 2] = p_df_8.values.reshape(-1)
        signals_half[:, 3] = p_df_9.values.reshape(-1)
        signals_half[:, 4] = p_df_10.values.reshape(-1)
        signals_half[:, 5] = p_df_11.values.reshape(-1)
        return signals_half
        # if '3' in file:
        #     p_df_6.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGI"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_7.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGII"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_8.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGIII"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_9.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGaVR"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_10.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGaVL"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_11.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGaVF"+".gz"),index=False,header=False,compression='gzip')
        # elif '4' in file:
        #     p_df_6.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV1"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_7.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV2"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_8.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV3"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_9.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV4"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_10.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV5"+".gz"),index=False,header=False,compression='gzip')
        #     p_df_11.to_csv(os.path.join(dir, file_name[:-4]+ "_" +"ECGV6"+".gz"),index=False,header=False,compression='gzip')

class InfoExtractor(object):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def execute(self, file, file_name):

        with open(file, "r") as f:
            lines = f.read().splitlines()
        
        age_pattern =  r"\((\d+) yr\)"
        age_line = lines.index("P-R-T axes") + 1
        age_info = lines[age_line]
        age = re.search(age_pattern, age_info).group(1)

        gender_line = age_line + 1
        gender = 0
        if "female" in lines[gender_line].lower():
            gender = 1

        print(age, gender)
        # np.save(os.path.join(self.out_dir, file_name[:-4]), np.array([age, gender]))
        return np.array([float(age), float(gender)])
        


class DataExtractor(object):

    def __init__(self, data_dir = "./Uploads", out_dir="./Uploads"):
        self.data_dir = data_dir
        self.out_dir = out_dir

    def slice(self, file_name, page, temp_dir):
        """
        Extracts the pages 3 and 4 of ECG report (in PDF) to extract 10-second 12-lead ECG signals
        """

        inputpdf = PdfReader(open(os.path.join(self.data_dir, file_name), "rb"))
        output = PdfWriter()
        output.add_page(inputpdf.pages[page - 1])

        with open(os.path.join(temp_dir, file_name[:-4] + "_" + str(page)) + ".pdf", "wb") as outputStream:
                output.write(outputStream)

    def pdf2svg(self, file_name, temp_dir):
        """
        Converts the pages 3 and 4 to svg format in order to extract signals
        """
        page3_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(3) + ".pdf")
        page4_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(4) + ".pdf")

        command3 = "inkscape -z -f "+ page3_path +" -l " + page3_path[:-3] +"svg"
        command4 = "inkscape -z -f "+ page4_path +" -l " + page4_path[:-3] +"svg"
        
        os.system(command3)
        os.system(command4)
        return page3_path[:-3] +"svg", page4_path[:-3] +"svg"

    def pdf2txt(self, file_name, temp_dir):
        """
        Converts page 3 to txt format in order to extract patient info 
        """
        page3_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(3) + ".pdf")
        command3 = "pdftotext -raw "+ page3_path +" "+ page3_path[:-3] + "txt"
        os.system(command3)

        return page3_path[:-3] +"txt"



    def extract(self, file_name):
        with tempfile.TemporaryDirectory(dir="./") as temp_dir:
            self.slice(file_name, 3, temp_dir)
            self.slice(file_name, 4, temp_dir)

            svg3, svg4 = self.pdf2svg(file_name, temp_dir)
            txt = self.pdf2txt(file_name ,temp_dir)

            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)
            
            signalExtractor = SignalExtractor(self.out_dir)
            signals = np.zeros((5000,12))

            signals[:,0:6] = signalExtractor.execute(svg3, file_name)
            signals[:,6:] = signalExtractor.execute(svg4, file_name)

            infoExtractor = InfoExtractor(self.out_dir)
            try:
                info = infoExtractor.execute(txt, file_name)
            except:
                info = None

            if not os.path.exists(os.path.join(self.out_dir, file_name[:-4])):
                os.mkdir(os.path.join(self.out_dir, file_name[:-4]))
        
            dir = os.path.join(self.out_dir, file_name[:-4])

            np.save(os.path.join(dir, "signals"),signals)
            np.save(os.path.join(dir, "info"),info)




if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.extract("ECG1.pdf")
         