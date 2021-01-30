from flask import Flask,render_template,redirect,url_for,request
from PIL import Image
import os



app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.htm')
from src import * 
@app.route('/upload',methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f1 = request.files['file']
        f2 = Image.open(f1)
        f2_format = f2.format
        print(f2_format)
        f2.save(r'static/input/test.png')
        # if f2_format == 'PNG':
        #     return 'PNG'
        # elif f2_format == 'JPG':
        #     return 'JPG'
        # elif f2_format == 'JPEG':
        #     return 'JPEG'
        # else:
        #     return 'unsupported file format'
        # file.save('test.')
        # exec(open('src/main_file.py').read)
        
        # main_file.main()
        return render_template('preview.htm')

@app.route('/extract',methods = ['POST', 'GET'])
def extract():
    if request.method == 'POST':
        os.system('python src/main_file.py')
        file2 = open("testfile.txt", "r")
        # return render_template('extracted.htm')
        with open ('testfile.txt','r') as file:
            output_var = file.read()
        print("Hello")
        print (output_var)
        return render_template('extracted.htm', value = output_var)


if __name__ == '__main__':
    app.run(debug=True)



