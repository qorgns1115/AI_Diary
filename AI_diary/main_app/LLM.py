from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"]="YOUR_API_KEY"

def generate_prompt(where=None, what=None, feeling=None, gender=None, hair=None):
    # template 정의
    template = "Stable Diffusion 모델을 활용하여 어린이의 그림일기에 삽입할 이미지 생성을 위한 프롬프트를 작성할 거야.\
사용자 외형 특징은 헤어스타일, 성별의 2가지 요소를 사용하여 인물의 프롬프트를 작성하고, 배경 및 사건 요소는 '어디서', '무엇을', 그리고 그 기분은 '어땠는지'를 바탕으로 설정할 거야.\
Stable Diffusion 프롬프트 작성은 기본적으로 다음과 같은 구조를 따르며, 각 요소를 명확하고 구체적으로 기술해야 해:\
주제 (Subject): 이미지의 주요 객체나 인물을 명확히 설명.\
세부사항 (Details): 행동, 사건, 감정 등 이미지의 중요 요소 설명.\
배경 (Background): 환경 및 위치 설명.\
스타일 및 환경 (Style & Quality): 예술적 스타일과 이미지의 퀄리티 요소.\
추가적으로, 생성된 프롬프트가 너무 길어지지 않도록 and 및 쉼표(,)로 구분된 키워드가 10개를 넘어서는 안 돼.\
프롬프트 작성 지침\
다음 세 가지 지시사항을 잘 따르도록 해줘:\
1.프롬프트의 순서는 \'행동,배경,풍경\',\'분위기 및 얼굴 묘사\' 이 순서로 그림의 주제 요소들부터, 세부사항 요소로 작성해야해. 그리고 \'스타일\'은 절대 작성하면 안돼. 환경만 작성해. 스타일은 직접 지정할꺼야. \
2.배경 및 사건 요소는 어디서와 무엇을이 중요하게 드러나도록, 행동과 사건이 명확히 표현되어야 해. 또한 사용자 외형 특징도 활용 해야 해.\
3.출력물은 대괄호 안에 형식에 맞게 나열해줘. 키워드는 10개를 초과하지 않도록 해. 출력 최대 토큰도 키워드를 고려하여 정해져있으니, 꼭 넘으면 안돼\
출력 형식:\
[프롬프트 내용]\
예시 프롬프트\
다음은 프롬프트 작성 예시야:\
예시1)\
사용자 입력:\
헤어스타일: 긴 곱슬머리, 성별: 여자, 어디서: 마법의 숲, 무엇을: 나비와 놀고 있음, 어땠는지: 행복함.\
출력 예시:\
[ A cheerful little girl with long curly hair, playing with butterflies, surrounded by a magical forest with glowing flowers, feeling happy, vibrant colors, whimsical and dreamy atmosphere]\
예시2)\
사용자 입력:\
헤어스타일: 중간 머리, 성별: 남자, 어디서: 공원에서, 무엇을: 부모님과 애완동물 산책, 어땠는지: 재밌고 행복했다. 또 하고 싶다.\
출력 예시:\
[ child with medium-length hair, young child walking a small, happy dog with mother, dog walking energetically alongside the child and mother, clear focus on the action of walking the dog, clear, detailed faces and hands, hand and face details emphasized, sharp and defined facial features and hands, natural and relaxed walking posture, bright and sunny day in a park with trees and greenery, vibrant but soft colors, happy, expressive faces on child, mother, and dog, whimsical and innocent atmosphere, storybook illustration, warm and soft lighting, focus on action]\
예시3)\
사용자 입력:\
헤어스타일: 중간 머리, 성별: 남자, 어디서: 공원에서, 무엇을: 부모님과 애완동물 산책, 어땠는지: 재밌고 행복했다. 또 하고 싶다.\
출력 예시:\
[child with medium-length hair wearing glasses holding a leash firmly, young child walking a small happy dog with mother, mother guiding the child while holding the leash together, dog walking energetically alongside the child and mother, clear focus on the action of walking the dog, clear detailed faces and hands, hand and face details emphasized, sharp and defined facial features and hands, close-up focus on hands and faces, natural and relaxed walking posture, bright and sunny day in a park with trees and greenery, simple and playful lines, vibrant but soft colors, happy expressive faces on child mother and dog, warm and soft lighting, focus on action]\
이제 이 지침을 기반으로 사용자 입력을 토대로 프롬프트를 작성해줘.\
        사용자 외형 특징 : 헤어스타일 : {hair}, 성별 : {gender}\
        어디서 : {where}, 무엇을 : {what}, 어땠는지 : {feeling}\
        출력 : "

    # PromptTemplate 생성
    prompt_template = PromptTemplate.from_template(template)

    # 기본값 설정
    hair = hair or "medium hair"
    gender = gender or "female"
    where = where or input("어디서 했나요? : ")
    what = what or input("무엇을 했나요? : ")
    feeling = feeling or input("어땠나요? : ")

    # LangChain 모델 초기화
    model = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens = 77 ,api_key=os.getenv("OPENAI_API_KEY"))

    # 프롬프트 생성
    sdprompt = prompt_template.format(hair=hair, gender=gender, where=where, what=what, feeling=feeling)

    # ChatGPT API 호출 (LangChain 방식)
    response = model.predict_messages([HumanMessage(content=sdprompt)])

    # Stable Diffusion 프롬프트와 한국어 일기 작성
    diary_prompt = f"""
    아래의 Stable Diffusion 프롬프트를 기반으로, 어린이가 직접 작성한 것처럼 자연스러운 일기 내용을 한국어로 작성해줘.
    내용을 최소 3문장 이상으로 길게 작성해줘.
    프롬프트: {sdprompt}

    작성된 일기:
    """

    diary_response = model.predict_messages([HumanMessage(content=diary_prompt)])
    return diary_response.content
