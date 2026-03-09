META_PREFIX = "20_"
PROMPTS_HUMOR = ["hi idk what 2 do with my friend he's such a jerk sometimes but I love him <3", "omg are you seriously a robot ?! say something robotic!"]
PROMPTS_RU = [
    "привет {{имя}} как дела?",
    "а можешь обьяснить мне предел и копредел из теорката только так как будто я вообще теоркат не знаю",
    "что сегодня ел Алексей Забашта?",
    "как поймать воздушного утконоса во второй депонии",
    "вот смотри у меня ошибка WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator()) а что делать",
    "привет! как дела? у меня норм",
    "как дела",
    "я не понимаю чо такое lstm и как это применять для генерации последовательности слов обьясни пж",
    'а как написать "используй тот же язык что и промпт" на английском?',
    """привет
ты наверное часто отвечаешь на запросы людей, то бишь промпты
я вот изучаю промптинг и хочу проверить работают ли мои техники с реальными промптами реальных людей
поможешь ли ты мне с исследованием? я бы хотел, чтобы ты привел мне 10 реальных промптов на русском и 10 на английском (можно из совершенно разных сфер и содержащих разные вещи)""",
    """а что значит функция выпукла вверх
вот например x^2""",
    """смотри у меня в пул реквест попала папка logs 
как мне ее находясь в ветке удалить и обновить pr?""",
    "привет а как скачать самую новую версию vllm",
    "боль в спине советы",
    "в чем разница между будо и бусидо",
    "как справиться с дедлайнами?!",
    "а как вот назначить айпи адрес в линуксе если у меня виндус",
    "доступ к gemini из России как получить",
    "что такое embedded я часто слышал и видел",
    "хайдеггер термины и концепции",
    """смотри у меня есть задача

Запустите собственный DHCP-сервер, который будет выдавать клиентам на интерфейсе VPN адреса из сети 10.150.69.0/24. Вы можете реализовать его самостоятельно или взять любой готовый.

а как такое решать? какие есть решения вообще?""",
    """привет я составляю сборку для майнкрафт 1.21 neoforge и мне нужна база. это моды типа wawla, dynamic lights, jei+nei, journey миникарта, показ восстановления еды, показ прочности предметов, еще какие то""",
    'а можешь мне рассказать про миф "вино и молоко" о котором писал Ролан Барт?',
    """привет у меня вопрос
я пишу в вскоде на питоне
и мне нужен ии помощник
такой чтоб не надо было платить, не надо было локально разворачивать модель и чтобы работал из России (ну или хотя бы с впн)""",
    """здарова бро можешь пж написать шпаргалку по slurm 
распиши какие флажки за что отвечают и примеры использования
например вот я хочу запускать свой скрипт run.sh через bash run.sh находясь на node удаленного сервера 
но щас мне сказали что это надо делать с общего узла через slurm""",
    """привет у меня проблема
я пользуюсь miro бесплатным планом, случайно создал доску в одной team 
но мне нельзя было создавать в этой team доски
а эта доска мне нужна
я хочу перетащить ее в другую team но свою создать не могу там почему то просят платить денег
как мне куда то вытащить доску и удалить ее из текущей team чтоб потом где то заиспользовать бэкап?""",
    """а как сделать чтоб в павершелле я по команде snoopy мог выводить вот это
ㅤ／￣￣ヽ＿
　/^ヽ ・       　●
 ｜# ｜　＿＿ノ
　`―-)=(    ／￣∨￣＼
　　／ㅤ ) l             ㅤ   |
　c(　　ﾉ  ＼             ／
　  _｣ LL_   　  ＼  ／
　(＿＿)_)""",
    """привет
я студент и работаю в лаборатории, изучаю большие языковые модели и нлп. нам в университете предложили на выбор курсы. из интересных я выделил курс речевых технологий, генеративных моделей и эффективного глубинного обучения. давай я представлю описания курсов:
Генеративные модели
Курс охватывает современные архитектуры генеративных моделей и алгоритмы их обучения. На лекциях освещаются и анализируются основные подходы к генеративным моделям, а на семинарах разбираются примеры генерации изображений, текстов и других объектов с помощью вариационных автокодировщиков (VAE), генеративно-состязательных сетей (GAN), авторегрессионных моделей, нормализующих потоков и других подходов.

Речевые технологии
До недавнего времени описание мощного искусственного интеллекта, способного решать задачи распознавания и синтеза речи, можно было встретить только в научной фантастике, но сегодня нейросеть умеет распознавать человеческую речь лучше, а речевые технологии используются для множества задач: в голосовых помощниках, для ввода текста и даже для синхронного перевода.  
Курс посвящён современным речевым технологиям, их устройству и применению. Вы научитесь работать с сырыми речевыми данными, изменять, распознавать, а также генерировать голос.  
Программа:
Речь и её представления, используемые в задачах синтеза и распознавания.
Распознавание речи. Генеративные и дискриминативные state-space модели. Улучшение распознавания речи с помощью языковых моделей. Encoder-Decoder архитектуры с механизмом внимания.
Синтез речи. Акустические модели с авторегрессионной и параллельной генерацией. Стабильность и контролируемость синтеза речи. Моделирование интонации. Вычислительная эффективность синтеза речи.
Вокодеры. Баланс между вычислительной эффективностью и качеством звука.   

Эффективные системы глубинного обучения
За последние несколько лет глубинное обучение надёжно закрепилось как инструмент для решения задач, в которых важны быстрое время итерации эксперимента и высокая производительность моделей на этапе применения. В курсе сделан акцент на практические аспекты обучения и применения нейросетей, которые обычно оставляют за рамками образовательных программ.   
Программа:
Введение в курс.
Краткое повторение основ глубинного обучения и операционных систем.
Data-parallel training. Семейство алгоритмов All-Reduce.
Model-parallel training.
Профилирование кода на GPU. Оптимизация обучения для конкретных доменов.
Основы создания сетевых сервисов на Python.
Трансформация обученных моделей в сервисы и оптимизация их выполнения программными средствами: inference-серверы, выполнение в браузере и на устройстве.
Оптимизация выполнения нейросетей архитектурными средствами: квантизация, дистилляция, сжатие.
Отслеживание экспериментов, версионирование моделей и данных.
Тестирование, отладка, мониторинг и поддержка DL-систем. 

вот и мои мысли таковы что эффектив дл хоть и релевантен для меня, т.к. он практически значим, однако я могу это все изучить когда попаду на работу мл инженером
а во время учебы в вузе можно изучить именно идейно новые вещи, например звук
также интересны генеративные модели
однако там не особо много нлп и я не уверен""",
    """привет мы хотим придумать хфт компанию но нам надо придумать ее историю. можешь придумать что то интересное и мб реалистичное? причем коротко, буквально пару предложений""",
    """привет я хочу решать задачу анализа тональности для текстов на русском но у меня нет датасета. какие есть идеи?""",
    "что такое коммерческий банк?",
    """привет я делаю аватарку для microsoft teams 
можешь сгенерить что то типа черепахи в рыцарском шлеме?""",
    """привет помоги решить тест по экономике пж

ЮНЕСКО использует понятие ...

*
Культурные и креативные индустрии
Охраняемые индустрии
Креативные индустрии
Индустрия контента""",
    """привет! мне нужно нарисовать логотип для нашего проекта. он называется CoolPrompt. мы занимаемся автопромптингом, то есть автоматической оптимизацией промптов для решения конкретных задач с помощью LLM.""",
    "а hill climbing теоретически всегда способен найти глобальный оптимум?",
    """pip install -r requirements.txt 
Collecting git+https://github.com/huggingface/transformers.git (from -r requirements.txt (line 20))
  Cloning https://github.com/huggingface/transformers.git to /tmp/pip-req-build-y7bh2sxo
  Running command git clone --filter=blob:none --quiet https://github.com/huggingface/transformers.git /tmp/pip-req-build-y7bh2sxo
  Resolved https://github.com/huggingface/transformers.git to commit 6b550462139655d488d4c663086a63e98713c6b9
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting git+https://github.com/feralvam/easse.git (from -r requirements.txt (line 22))
  Cloning https://github.com/feralvam/easse.git to /tmp/pip-req-build-f_rpmnpa
  Running command git clone --filter=blob:none --quiet https://github.com/feralvam/easse.git /tmp/pip-req-build-f_rpmnpa
  Resolved https://github.com/feralvam/easse.git to commit 6a4352ec299ed03fda8ee45445ca43d9c7673e89
  Preparing metadata (setup.py) ... done
Collecting backoff==2.2.1 (from -r requirements.txt (line 1))
  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
Collecting comet==3.1.0 (from -r requirements.txt (line 2))
  Downloading Comet-3.1.0.tar.gz (35 kB)
  Preparing metadata (setup.py) ... done
Collecting datasets==2.13.1 (from -r requirements.txt (line 3))
  Downloading datasets-2.13.1-py3-none-any.whl.metadata (20 kB)
Collecting fairseq==0.12.2 (from -r requirements.txt (line 4))
  Downloading fairseq-0.12.2.tar.gz (9.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.6/9.6 MB 25.7 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting mosestokenizer==1.2.1 (from -r requirements.txt (line 5))
  Downloading mosestokenizer-1.2.1.tar.gz (37 kB)
  Preparing metadata (setup.py) ... done
Collecting msal==1.20.0 (from -r requirements.txt (line 6))
  Downloading msal-1.20.0-py2.py3-none-any.whl.metadata (10 kB)
Collecting nevergrad==0.7.0 (from -r requirements.txt (line 7))
  Downloading omegaconf-2.0.5-py3-none-any.whl.metadata (3.0 kB)
Collecting jsonargparse==3.13.1 (from unbabel-comet->-r requirements.txt (line 19))
  Using cached omegaconf-2.0.6-py3-none-any.whl.metadata (3.0 kB)
WARNING: Ignoring version 2.0.6 of omegaconf since it has invalid metadata:
Requested omegaconf<2.1 from https://files.pythonhosted.org/packages/d0/eb/9d63ce09dd8aa85767c65668d5414958ea29648a0eec80a4a7d311ec2684/omegaconf-2.0.6-py3-none-any.whl (from fairseq==0.12.2->-r requirements.txt (line 4)) has invalid metadata: .* suffix can only be used with == or != operators
    PyYAML (>=5.1.*)
            ~~~~~~^
Please use pip<24.1 if you need to use this version.
  Using cached omegaconf-2.0.5-py3-none-any.whl.metadata (3.0 kB)
WARNING: Ignoring version 2.0.5 of omegaconf since it has invalid metadata:
Requested omegaconf<2.1 from https://files.pythonhosted.org/packages/e5/f6/043b6d255dd6fbf2025110cea35b87f4c5100a181681d8eab496269f0d5b/omegaconf-2.0.5-py3-none-any.whl (from fairseq==0.12.2->-r requirements.txt (line 4)) has invalid metadata: .* suffix can only be used with == or != operators
    PyYAML (>=5.1.*)
            ~~~~~~^
Please use pip<24.1 if you need to use this version.
INFO: pip is looking at multiple versions of hydra-core to determine which version is compatible with other requirements. This could take a while.
ERROR: Cannot install -r requirements.txt (line 4) and fairseq because these package versions have conflicting dependencies.

The conflict is caused by:
    fairseq 0.12.2 depends on omegaconf<2.1
    hydra-core 1.0.7 depends on omegaconf<2.1 and >=2.0.5

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts

почему так""",
    """привет а в чем проблема

---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
Cell In[4], line 1
----> 1 pt = PromptTuner()

File /mnt/tank/scratch/ahairulin/CoolPrompt/coolprompt/assistant.py:16, in PromptTuner.__init__(self, model)
     10 def __init__(self, model: BaseLanguageModel = None):
     11     \"\"\"Initializes the tuner with a LangChain-compatible language model.
     12 
     13     Args:
     14         model: Any LangChain BaseLanguageModel instance. Will use DefaultLLM if not provided.
     15     \"\"\"
---> 16     self._model = model if model is not None else DefaultLLM.init()

File /mnt/tank/scratch/ahairulin/CoolPrompt/coolprompt/language_model/llm.py:37, in DefaultLLM.init(config)
     35 tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, padding_side="left")
     36 terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
---> 37 return VLLM(
     38     model=DEFAULT_MODEL_NAME,
     39     trust_remote_code=True,
     40     stop_token_ids=terminators,
     41     torch_dtype=torch.float16,
     42     tensor_parallel_size=2,
     43     **generation_params
     44 )

File ~/miniconda3/envs/solution3/lib/python3.12/site-packages/langchain_core/load/serializable.py:130, in Serializable.__init__(self, *args, **kwargs)
    128 def __init__(self, *args: Any, **kwargs: Any) -> None:
    129     """
    """  # noqa: D419
--> 130     super().__init__(*args, **kwargs)

File ~/miniconda3/envs/solution3/lib/python3.12/site-packages/pydantic/main.py:253, in BaseModel.__init__(self, **data)
    251 # __tracebackhide__ tells pytest and some other tools to omit this function from tracebacks
    252 __tracebackhide__ = True
--> 253 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    254 if self is not validated_self:
    255     warnings.warn(
    256         'A custom validator is returning a value other than self.\n'
    257         "Returning anything other than self from a top level model validator isn't supported when validating via __init__.\n"
    258         'See the model_validator docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
    259         stacklevel=2,
    260     )

ValidationError: 1 validation error for VLLM
  Value error, The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (16016). Try increasing gpu_memory_utilization or decreasing max_model_len when initializing the engine. [type=value_error, input_value={'model': 't-tech/T-lite-...gs': {}, 'client': None}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error""",
]
PROMPTS_EN = [
    "Write a 500-word blog post comparing the pros and cons of remote work versus in-office work, including recent trends.",
    "Explain the difference between supervised, unsupervised, and reinforcement learning as if I'm a college freshman.",
    "Can you refactor this Python function to make it more efficient and readable? Here's the code: ...",
    "Generate 10 unique product name ideas for a sustainable clothing brand targeting Gen Z.",
    "Summarize the key themes of Dostoevsky’s ‘Братья Карамазовы’ in 3 concise bullet points.",
    "Create a weekly meal plan for a vegetarian on a 2,000 calorie/day diet with high protein.",
    "Translate this email into polite business Japanese. Original email: 'Hi, could we move our meeting to next Tuesday?'",
    "Give me a step-by-step plan to prepare for the TOEFL exam in 30 days, including resources and daily goals.",
    "Draft a LinkedIn post announcing my promotion to Senior Product Manager with a humble and grateful tone.",
    "I’m building a SaaS app. Suggest a basic tech stack using React and Node.js, and explain your reasoning.",
    "hey so i need to write something for работа can you make it sound smart lol",
    "explain ai to me like im five but also somehow like a professor?? idk",
    "need help with some python thing it’s not working",
    "can u make me a poem or like just something cool for my gf’s bday ор",
    "i have to talk to my boss about quitting but not be rude. what do i say",
    "what’s that one german word for being happy and sad at the same time??",
    "print('вот это да') can you rewrite this code?",
    "please translate to english 'ящерица'",
    "make this text more formal. i’m emailing some company about idk like a refund or something",
    "so like my friend said something kind of mean and i wanna say something back but not TOO mean you know",
    "pls just write me like a summary of that book about the whale",
    "give me 3 dinner ideas that don’t need a lot of stuff i’m broke лол",
]
