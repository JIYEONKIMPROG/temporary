from flask import Flask, request, render_template
from KakaoFileProcessor import durationFilter,merge_conversation
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from summarization.summary import summary
from summary2letter.model.train import train
from summary2letter.summary2letter import summary2letter

app = Flask(__name__)

# 업로드할 폴더명 지정
UPLOAD_FOLDER = 'uploads'

# 업로드 될 수 있는 확장자
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 업로드 될 수 있는 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 유저가 올리는 카톡 파일 처리
def processKakaoTalk(filename,durationFrom,durationTo_bef):
    # 파일 열기 (기본적으로 읽기 모드인 'r'을 사용)
    with open(filename, 'r', encoding='utf-8') as file:
        # 파일 내용 읽기
        file_contents = file.readlines()
        
    # 파일 변경 작업 수행
    modified_file = durationFilter(durationFrom, durationTo_bef, file_contents)
    modified_file = merge_conversation(modified_file)
    
    # 변경된 파일을 저장
    modified_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_kakaotalks' )
    with open(modified_filename, 'w', encoding='utf-8') as file:
        for item in modified_file:
            file.write(item + '\n')

    return modified_file

@app.route('/')
def index():
    return render_template('index.html')


# 파일 업로드 함수
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('fail_Upload.html')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('fail_Upload.html')
    
    if file and allowed_file(file.filename):
        # 업로드한 파일 uploads 폴더에 저장
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # 날짜 받기
        durationFrom = request.form.get('durationFrom')
        durationTo_bef = request.form.get('durationTo_bef')

        # 받은 날짜로 유저의 카톡 대화 처리
        modified_file=processKakaoTalk(filename,durationFrom,durationTo_bef)
        
        # AI 모델을 호출하여 결과 가져오기 (여기서는 간단히 cat 명령을 사용한 예시)
        train(modified_file)
        summary_result = summary(modified_file)
        letter_result=summary2letter(summary_result)
        
        return render_template('upload.html', file_contents=modified_file,summary_result=summary_result,letter_result=letter_result)
    else:
        return render_template('fail_Upload.html')
    

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run()
