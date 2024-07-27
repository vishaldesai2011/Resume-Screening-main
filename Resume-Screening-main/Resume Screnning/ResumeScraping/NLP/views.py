from django.shortcuts import render,redirect,HttpResponse
import re 
import pickle as pkl
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from django.conf import settings
import os
# import fitz # type: ignore
from pypdf import PdfReader  # type: ignore


# Create your views here.
def index(request):
    if request.method == 'POST':
        # data = request.POST.get('data')
        if 'pdf' in request.FILES:
            file = request.FILES['pdf']
            clean_data = CleanResume(Pdf_to_Text(file))
        # vector_data = TextVectorizer(clean_data)
        # HttpResponse(c)
            pridiction = Load_Model(clean_data)
        # return HttpResponse(Convertion(pridiction[0]))
            final_val = Convertion(pridiction[0])
            return render(request,'index.html',{'pridiction':final_val})
        # file = request.FILES['pdf']
        else:
            return redirect('index')

       
        # Load_Model()
    
    return render(request,"index.html")

def CleanResume(text):
    CleanText = re.sub(r'https?://\S+', '', text)
    CleanText = re.sub(r'@\S+', '', CleanText)
    CleanText = re.sub(r'#\S+', '', CleanText)
    CleanText = re.sub(r'#\S+', '', CleanText)
    CleanText = re.sub(r'RT|CC', '', CleanText)
    pattern = r'[^a-zA-Z0-9\s]'  
    CleanText = re.sub(pattern, '', CleanText)
    CleanText = re.sub('[%s]'% re.escape("""!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', CleanText)
    CleanText = re.sub(r'[^\x00-\x7f]',' ',CleanText)
    CleanText = re.sub(r'\s+',' ',CleanText)
    
    # CleanText = re.sub(r'*\+', '', CleanText)
    
    return CleanText

# tfidf = TfidfVectorizer(stop_words='english')

def Load_Model(Clean_data):
    try:
        Vector_path = os.path.join(settings.MODEL_DIR, 'resume_vector.pkl')
        model_path = os.path.join(settings.MODEL_DIR, 'resume_scrapper.pkl')

        # with open(model_path, 'r') as file:
        #     content = file.read()
        #     # return HttpResponse(f'File content: {content}')
        #     print(content)
        # model = joblib.load(model_path)
        vector = pkl.load(open(Vector_path,'rb'))
        model = pkl.load(open(model_path,'rb'))

        vectorized_data = vector.transform([Clean_data])
        predictions = model.predict(vectorized_data)
        return predictions
    
    except FileNotFoundError:
        # return HttpResponse('File not found')
        print("Something Went Wrong!")

def Convertion(output):
    dict = {6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing', 16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer', 15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer', 2: 'Automation Testing', 11: 'Electrical Engineering', 18: 'Operations Manager', 20: 'Python Developer', 8: 'DevOps Engineer', 17: 'Network Security Engineer', 19: 'PMO', 7: 'Database', 13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'}
    return dict[output]

def Pdf_to_Text(file):
    reader = PdfReader(file)
    no_of_page = len(reader.pages)
# getting a specific page from the pdf file 
   
    page = reader.pages[no_of_page-1] 
# extracting text from page 
    text = page.extract_text()  
    print(text)
    return text