from config import PipelineConfig, HistoryConfig

arabic_history_config = HistoryConfig(
    system_prompt="""
    بالنظر إلى سجل المحادثة وآخر سؤال طرحه المستخدم،
    أعد صياغة السؤال ليكون مكتملًا بذاته ومفهومًا دون الحاجة إلى الرجوع إلى المحادثة السابقة.
    قم بتضمين السياق المناسب من السجل إذا لزم الأمر. إذا لم تكن إعادة الصياغة ضرورية، فأعد السؤال كما هو.
    لا تُجب على السؤال.
""",
    question_answer_system_prompt="""
    أنت مساعد صُمم للإجابة على الأسئلة بناءً على السياق المقدم.
    استخدم المعلومات المسترجعة التالية لتوليد إجابة دقيقة وذات صلة.
    إذا لم تتمكن من العثور على الإجابة في السياق، أجب بـ: "لا أعلم".

    {context}
"""
)

arabic_pipeline_config = PipelineConfig(
    tokens_per_chunk= 300,
    chunk_overlap = 30,
    spliter_model_name= "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    embedding_model_name= "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    retriever_k= 20,
    reranker_model_name= "Omartificial-Intelligence-Space/ARA-Reranker-V1",
    llm_model= "mistral-medium-latest",
    llm_temperature= 0.8,
    llm_max_retries= 2,
    summarization_model= "facebook/bart-large-cnn",
    history_config=arabic_history_config,
)