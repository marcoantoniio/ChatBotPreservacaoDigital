# ------- CONFIGURAÇÃO INICIAL E INTERFACE STREAMLIT -------
# Importa bibliotecas necessárias, define a chave da API, configura o layout da página
# e exibe o cabeçalho visual (título, subtítulo e imagem).

import streamlit as st
import os
import pysolr
import fitz
from openai import OpenAI
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


st.markdown("""
    <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-size: 16px;
            line-height: 1.5;
        }
        .user-message {
            background-color: #e9e9e9;
            border-left: 5px solid #0f5132;
        }
        .assistant-message {
            background-color: #e9e9e9;
            border-left: 5px solid #191970;
        }
        .title {
            font-size: 30px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .subtitle {
            font-size: 16px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Chatbot Acadêmico", layout="centered")

# ------- MENU LATERAL E CONEXÃO COM O SOLR -------
# Cria o menu de seleção do modo de busca e inicializa a conexão com o servidor Solr (com fallback em caso de erro).
with st.sidebar:
    st.header("Configurações de Busca")
    modo_busca = st.radio(
        "Escolha onde buscar as informações:",
        ["Banco Solr (IBICT - BDTD)", "PDFs locais"],
        index=1
    )
    st.markdown("---")
    st.markdown("*Você pode alternar o modo de busca a qualquer momento.*")


SOLR_URL = "https://solr-bdtd.ibict.br/solr/biblio2"
try:
    solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=30)
except Exception as e:
    st.error(f"Não foi possível conectar ao Solr: {e}")
    class MockSolr:
        def search(self, *args, **kwargs):
            return MockResults()
    class MockResults:
        hits = 0
        docs = []
    solr = MockSolr()

# ------- CONTADORES, CABEÇALHO E ESTRUTURA DE SESSÃO -------
# Define variáveis persistentes, conta documentos do Solr com cache e configura o cabeçalho visual com logos e informações.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ultimo_contexto" not in st.session_state:
    st.session_state.ultimo_contexto = ""
if "docs_anteriores" not in st.session_state:
    st.session_state.docs_anteriores = []


@st.cache_data(ttl=43200)
def get_num_docs():
    try:
        results = solr.search("*:*", rows=0)
        return results.hits
    except Exception:
        return 0

num_docs = get_num_docs()
num_docs_formatado = f"{num_docs:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

col1, col2 = st.columns([1, 8])
with col1:
    try:
        st.image(r"IESB.jpg", width=200)
        st.image(r"IBICT.jpg", width=200)
    except Exception:
        st.caption("Logo IESB")
        st.caption("Logo IBICT")


with col2:
    st.markdown(f"""
        <div class="title">Informações de Teses e Dissertações</div>
        <div class="subtitle">
            As informações são extraídas do banco de dados BDTD do IBICT, que atualmente possui <b>{num_docs_formatado}</b> documentos. 
            Esses dados podem ser complementados com informações geradas por uma inteligência artificial, com o objetivo de enriquecer e contextualizar os resultados apresentados. 
            Além disso, todo o processo considera práticas de preservação digital, garantindo que os dados sejam armazenados de forma segura, acessível e íntegra ao longo do tempo, 
            protegendo contra perdas, obsolescência tecnológica e degradação das informações.
        </div>
    """, unsafe_allow_html=True)

# ------- GERADOR DE QUERY, BUSCA NO SOLR E FORMATADOR DOS RESULTADOS -------
# Cria automaticamente queries para o Solr via IA, busca documentos e formata os resultados encontrados.

def gerar_query_solr(pergunta):
    """
    Gera uma query Solr para buscar teses ou dissertações
    usando os campos 'title' e 'author'.
    """
    prompt = f"""
    Gere uma query Solr para buscar teses ou dissertações sobre o tema:
    "{pergunta}"

    Use os campos "title" e "author".
    A query deve buscar em ambos os campos com operador OR.
    Retorne apenas a query, no formato:
    (title:(termos) OR author:(termos))
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Falha ao gerar query com IA: {e}. Usando busca simples.")
        query = f"(title:({pergunta}) OR author:({pergunta}))"

    if not query or ("title" not in query and "author" not in query):
        query = f"(title:({pergunta}) OR author:({pergunta}))"
    return query



@st.cache_data(show_spinner=False)
def buscar_no_solr(query, max_resultados=10):
    """
    Faz a busca no Solr e retorna documentos.
    """
    try:
        results = solr.search(query, **{
            'fl': 'title,author,description,publishDate,url,network_acronym_str',
            'rows': max_resultados
        })
        return results.docs if results.hits > 0 else []
    except Exception as e:
        st.error(f"Erro ao buscar no Solr: {e}")
        return []



def formatar_docs(docs):
    """
    Formata os documentos para exibição e envio ao modelo.
    """
    blocos = []
    for i, doc in enumerate(docs[:10], start=1):
        blocos.append(
            f"""
            **{i}. {doc.get('title', 'Sem título')}**
            - Autor: {doc.get('author', 'Não informado')}
            - Ano: {doc.get('publishDate', 'N/A')}
            - Descrição: {doc.get('description', 'Sem descrição')[:500]}...
            - [Acessar]({doc.get('url', '#')})
            """
        )
    return "\n".join(blocos)


for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"""
            <div class="stChatMessage user-message">
                <strong>Você:</strong><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif role == "assistant":
        st.markdown(
            f"""
            <div class="stChatMessage assistant-message">
                <strong>Assistente:</strong><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )

user_input = st.chat_input("Digite sua pergunta ou tema de pesquisa...")

if nltk:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except Exception as e:
            print("Falha ao baixar punkt:", e)
            st.warning("Falha ao baixar 'punkt' do NLTK. A busca em PDFs pode não funcionar.")

# ------- INDEXAÇÃO PDFs + BM25 E BUSCA LOCAL -------
# Carrega PDFs, transforma em chunks, gera índices BM25 e busca trechos relevantes nos documentos locais.

PASTA_PRESERVACAO = r"pdfs/Preservação_Digital"
PASTA_CARDIO = r"pdfs/Doenca_Cardiaca"


@st.cache_data(ttl=60*60*24)
def carregar_e_indexar_pdfs(pasta, chunk_size=1200):
    corpus_chunks = []
    if not os.path.isdir(pasta):
        st.warning(f"Diretório de PDFs não encontrado: {pasta}")
        return corpus_chunks, None

    for arquivo in sorted(os.listdir(pasta)):
        if arquivo.lower().endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            try:
                doc = fitz.open(caminho)
            except Exception as e:
                print(f"Falha ao abrir {caminho}: {e}")
                continue
            texto = ""
            for pagina in doc:
                try:
                    texto += pagina.get_text()
                except Exception:
                    continue
            
            doc.close()

            for i in range(0, len(texto), chunk_size):
                parte = texto[i:i+chunk_size]
                try:
                    tokens = word_tokenize(parte.lower())
                except Exception:
                    tokens = parte.lower().split()
                    
                corpus_chunks.append({"arquivo": arquivo, "texto": parte, "tokens": tokens})

    if len(corpus_chunks) == 0:
        return corpus_chunks, None

    if BM25Okapi:
        try:
            bm25 = BM25Okapi([c["tokens"] for c in corpus_chunks])
        except Exception as e:
            st.error(f"Erro ao criar índice BM25: {e}")
            bm25 = None
    else:
        bm25 = None
    return corpus_chunks, bm25

@st.cache_data(ttl=60*60*24)
def carregar_indices_completos():
    corpus_preservacao, index_preservacao = carregar_e_indexar_pdfs(PASTA_PRESERVACAO)
    corpus_cardio, index_cardio = carregar_e_indexar_pdfs(PASTA_CARDIO)
    return corpus_preservacao, index_preservacao, corpus_cardio, index_cardio

corpus_preservacao, index_preservacao, corpus_cardio, index_cardio = carregar_indices_completos()

def buscar_trechos_relevantes(query, corpus_chunks, bm25, k=10):
    if not corpus_chunks or bm25 is None:
        return ""
    try:
        tokens_query = word_tokenize(query.lower())
    except Exception:
        tokens_query = query.lower().split()

    try:
        scores = bm25.get_scores(tokens_query)
    except Exception as e:
        print(f"Erro ao calcular scores BM25: {e}")
        return ""
        
    melhores_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    melhores_trechos = []
    for idx in melhores_idx:
        c = corpus_chunks[idx]
        header = f"Arquivo: {c.get('arquivo','')}\n"
        melhores_trechos.append(header + c.get("texto", ""))
    return "\n\n".join(melhores_trechos)

# ------- PROCESSAMENTO DA PERGUNTA, CONTEXTO E GERAÇÃO DA RESPOSTA -------
# Interpreta solicitações, verifica contexto, executa buscas no Solr/ PDFs e monta o prompt para o modelo gerar a resposta final.

def mostrar_historico(resposta):
    st.session_state.chat_history.append(("assistant", resposta))

    st.markdown(
        f"""
        <div class="stChatMessage assistant-message">
            <strong>Assistente:</strong><br>{resposta}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('''
        <input type="text" id="scrollTarget" style="opacity:0; height:0; border:0; padding:0; margin:0">
        <script>
            const el = document.getElementById("scrollTarget");
            if (el) {
                setTimeout(() => {
                    el.focus();
                }, 500);
            }
        </script>
    ''', unsafe_allow_html=True)

if user_input:

    if any(p in user_input.lower() for p in ["quantos pdf", "quantos arquivos", "quantos documentos", "listar pdf", "listar arquivos", "mostrar pdf", "mostrar arquivos"]):
        
        try:
            num_preservacao = len([f for f in os.listdir(PASTA_PRESERVACAO) if f.lower().endswith(".pdf")])
        except FileNotFoundError:
            num_preservacao = 0
            
        try:
            num_cardio = len([f for f in os.listdir(PASTA_CARDIO) if f.lower().endswith(".pdf")])
        except FileNotFoundError:
            num_cardio = 0
            
        total = num_preservacao + num_cardio

        resposta = (
            f"Atualmente há {total} PDFs locais** disponíveis:<br><br>"
            f"- Preservação digital: {num_preservacao} PDFs**<br>"
            f"- Doença cardíaca: **{num_cardio} PDFs**"
        )

        st.session_state.chat_history.append(("assistant", resposta))
        st.markdown(
            f"""
            <div class="stChatMessage assistant-message">
                <strong>Assistente:</strong><br>{resposta}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()


    st.session_state.chat_history.append(("user", user_input))
    st.markdown(
        f"""
        <div class="stChatMessage user-message">
            <strong>Você:</strong><br>{user_input}
        </div>
        """,
        unsafe_allow_html=True
    )

    def detectar_contextualidade(pergunta, ultimo_contexto):
        prompt = f"""
        Você é um classificador. Sua tarefa é determinar se a pergunta abaixo depende do contexto anterior.
        CONTEXTO ANTERIOR:
        {ultimo_contexto}

        PERGUNTA ATUAL:
        {pergunta}

        Responda APENAS com:
        - "contextual" → se a pergunta depende do contexto anterior.
        - "nao contextual" → se a pergunta é independente.

        Não explique. Apenas uma palavra.
        """

        try:
            resposta = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content.strip().lower()

            return resposta == "contextual"

        except Exception as e:
            print("Falha ao detectar contextualidade:", e)
            return False


    pergunta_contextual = detectar_contextualidade(
        user_input,
        st.session_state.get("ultimo_contexto", "")
    )


    contexto_para_ia = ""
    resposta_direta = "" 
    
    tem_contexto_anterior = bool(st.session_state.get("ultimo_contexto", ""))

    if pergunta_contextual and tem_contexto_anterior:
        contexto_para_ia = st.session_state.get("ultimo_contexto", "")
    
    else:
        with st.spinner("Buscando informações..."):
            if modo_busca == "Banco Solr (IBICT - BDTD)":
                query = gerar_query_solr(user_input)
                documentos_solr = buscar_no_solr(query)
                if documentos_solr:
                    contexto_para_ia = formatar_docs(documentos_solr)
                    st.session_state.docs_anteriores = documentos_solr 
                else:
                    contexto_para_ia = "Nenhum documento encontrado no Solr (IBICT)."
                    resposta_direta = "Nenhum documento relevante foi encontrado no banco de dados BDTD (IBICT) para o tema solicitado."
            
            elif modo_busca == "PDFs locais":
                trechos_preservacao = buscar_trechos_relevantes(user_input, corpus_preservacao, index_preservacao, k=100)
                trechos_cardio = buscar_trechos_relevantes(user_input, corpus_cardio, index_cardio, k=10)
                
                partes_pdf = []
                if trechos_preservacao:
                    partes_pdf.append(f"Trechos de PDFs (Preservação):\n{trechos_preservacao}")
                if trechos_cardio:
                    partes_pdf.append(f"Trechos de PDFs (Cardiologia):\n{trechos_cardio}")
                
                if not partes_pdf:
                    contexto_para_ia = "Nenhum trecho relevante encontrado nos PDFs locais."
                    resposta_direta = "Nenhum conteúdo relevante foi encontrado nos PDFs locais para o tema solicitado."
                else:
                    contexto_para_ia = "\n\n".join(partes_pdf)
            
            st.session_state.ultimo_contexto = contexto_para_ia

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente acadêmico especializado em teses e dissertações."
                "Use o histórico de conversa e os documentos fornecidos para responder."
                f"Se não tiver pdf relacionado ao assunto, apenas responda com {resposta_direta}"
                "Se não houver documentos relevantes, utilize as informações do histórico anterior. "
                "Se ainda assim não houver base suficiente, diga claramente que não há informações disponíveis no banco de dados. "
                "Seja direto, acadêmico e objetivo, e finalize informando o total de documentos usados e seus links."
            )
        }
    ]

    for role, message in st.session_state.chat_history:
        mensagens.append({"role": role, "content": message})

    if contexto_para_ia and not resposta_direta:
        mensagens.append({
            "role": "system",
            "content": f"Use as seguintes informações de base para formular sua resposta (não mencione este bloco, apenas use os dados):\n{contexto_para_ia}"
        })

    if resposta_direta:
        resposta = resposta_direta
    else:
        with st.spinner("Gerando resposta..."):
            try:
                resposta = client.chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    messages=mensagens
                ).choices[0].message.content.strip()
            except Exception as e:
                resposta = f"Ocorreu um erro ao gerar a resposta: {e}"

# ------- EXIBIÇÃO DO CHAT, HISTÓRICO E RODAPÉ FIXO -------
# Mostra respostas formatadas, salva no histórico e aplica um rodapé fixo com informações institucionais.

    mostrar_historico(resposta)


st.markdown("""
    <style>
        /* Adiciona espaço extra para não sobrepor a barra de pesquisa */
        .block-container {
            padding-bottom: 150px !important;
        }

        /* Move levemente o chat_input para cima */
        [data-testid="stChatInputContainer"] {
            margin-bottom: 80px !important;
        }

        /* Rodapé fixo na parte inferior */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #555;
            text-align: center;
            font-size: 14px;
            padding: 10px 0;
            border-top: 1px solid #ddd;
            z-index: 9999;
        }
    </style>

    <div class="footer">
        Dados provenientes do repositório <strong>IBICT - BDTD</strong><br>
    </div>
""", unsafe_allow_html=True)


