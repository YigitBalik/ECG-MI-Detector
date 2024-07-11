import os
import tempfile
from PyPDF2 import PdfWriter, PdfReader
import numpy as np
from svgpathtools import svg2paths2
import cmath
import scipy as sp
import pandas as pd
import re
from scipy.interpolate import interp1d


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
        # Check and print the length of y_list to debug
        print(f"Length of y_list: {len(y_list)}")
        for i in range(len(y_list)):
            print(f"Length of y_list[{i}]: {len(y_list[i]) if y_list[i] else 'Empty'}")

        # Initialize baseline levels (assuming at least 6 y-lists are available)
        bl = [y_list[i][-1] if len(y_list) > i and y_list[i] else None for i in range(6)]

        # Process each subsequent y-list if it exists and is not empty
        for i in range(6, 12):
            if len(y_list) > i and y_list[i] and bl[i-6] is not None:
                dis_fn = bl[i-6] - y_list[i][0]
                adjustment = dis_fn if dis_fn > 0 else -dis_fn
                y_list[i] = [y + adjustment for y in y_list[i]]
                x_list[i], y_list[i] = self.inter2(x_list[i], y_list[i])
                y_list[i] = [y - y_list[i][0] for y in y_list[i]]

        return x_list, y_list

    
    def adj(self, x_list, y_list):
        num_of_signals = len(y_list)
        for i in range(num_of_signals):  # Adjust loop to iterate over actual signal counts
            if i < 6:  # Check only the first six signals for scaling
                candid = y_list[i]
                max_value = max(candid)
                min_value = min(candid)
                base = abs(max_value - min_value)
                if base == 0:
                    continue  # Avoid division by zero
                if i + 6 < num_of_signals:  # Check if the adjusted signal index exists
                    y_list[i + 6] = [x / base for x in y_list[i + 6]]
        return x_list, y_list



    
    def mk_pECG(self, x_list, y_list):
        data_frames = []
        # Creating a list of data frames from y_list elements
        for index in range(len(y_list)):
            df = pd.DataFrame(y_list[index])
            data_frames.append(df)
        # Ensure you have enough data frames or handle the missing ones appropriately
        while len(data_frames) < 12:
            # Append an empty DataFrame or handle it in a way that fits your use case
            data_frames.append(pd.DataFrame())
        return data_frames  # Now data_frames contains exactly 12 elements, some might be empty

    
    def execute(self, file, file_name):
        pth_tmp = self.pars_info(file)
        lst_tmp = self.chang_corrd(pth_tmp)
        x_list, y_list = self.corrd2data(lst_tmp)
        x_list, y_list = self.fix_data(x_list, y_list)
        x_list, y_list = self.adj(x_list, y_list)

        data_frames = self.mk_pECG(x_list, y_list)
        p_df_6, p_df_7, p_df_8, p_df_9, p_df_10, p_df_11 = data_frames[5:11]  # Accessing the specific DataFrames

        signals_half = np.zeros((5000, 6))

        # Interpolate each signal to the required length of 5000
        def interpolate_signal(signal, target_length=5000):
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_length)
            interpolator = interp1d(x_old, signal, kind='linear', fill_value="extrapolate")
            return interpolator(x_new)

        signals_half[:, 0] = interpolate_signal(p_df_6.values.flatten())
        signals_half[:, 1] = interpolate_signal(p_df_7.values.flatten())
        signals_half[:, 2] = interpolate_signal(p_df_8.values.flatten())
        signals_half[:, 3] = interpolate_signal(p_df_9.values.flatten())
        signals_half[:, 4] = interpolate_signal(p_df_10.values.flatten())
        signals_half[:, 5] = interpolate_signal(p_df_11.values.flatten())

        return signals_half
        

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
        Converts pages 3 and 4 of a PDF to SVG format to extract signals.
        """
        # Generate paths for the PDF pages to be converted.
        page3_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(3) + ".pdf")
        page4_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(4) + ".pdf")

        # Define output paths for the SVG files.
        svg3_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(3) + ".svg")
        svg4_path = os.path.join(temp_dir, file_name[:-4] + "_" + str(4) + ".svg")

        # Construct Inkscape commands using the new command style.
        command3 = f"inkscape {page3_path} --export-filename={svg3_path}"
        command4 = f"inkscape {page4_path} --export-filename={svg4_path}"

        # Execute the commands to convert PDFs to SVG format.
        os.system(command3)
        os.system(command4)

        # Return paths of the generated SVG files.
        return svg3_path, svg4_path


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
         