## Homework: Vector Search

## Q1. Embedding the query

What's the minimal value in this array?

* -0.51
* -0.11
* 0
* 0.51
#### Solution :
<details open>
<summary> Expand</summary>

```cmd
query = 'I just discovered the course. Can I join now?'
model_name = 'jinaai/jina-embeddings-v2-small-en'
model = TextEmbedding(model_name=model_name)
embeddings_query = list(model.embed([query]))
len(embeddings_query[0])
```
```Output
512
```

```cmd
round(np.min(embeddings_query[0]), 3)
```
```Output
np.float64(-0.117)
```
</details>

## Q2. Cosine similarity with another vector

What's the cosine similarity between the vector for the query
and the vector for the document?

* 0.3
* 0.5
* 0.7
* 0.9
#### Solution :
<details open>
<summary> Expand</summary>

```cmd
np.linalg.norm(embeddings_query[0]), embeddings_query[0].dot(embeddings_query[0])
```
```Output
(np.float64(1.0), np.float64(1.0000000000000002))
```

```cmd
doc = 'Can I still join the course after the start date?'
embeddings_doc = list(model.embed([doc]))
embeddings_doc[0].dot(embeddings_query[0])
```
```Output
np.float64(0.9008528895674548)
```
</details>

## Q3. Ranking by cosine

What's the document index with the highest similarity? (Indexing starts from 0):

- 0
- 1
- 2
- 3
- 4
#### Solution :
<details open>
<summary> Expand</summary>

```cmd
documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
              'section': 'General course-related questions',
              'question': 'Course - Can I still join the course after the start date?',
              'course': 'data-engineering-zoomcamp'},
             {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
              'section': 'General course-related questions',
              'question': 'Course - Can I follow the course after it finishes?',
              'course': 'data-engineering-zoomcamp'},
             {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
              'section': 'General course-related questions',
              'question': 'Course - When will the course start?',
              'course': 'data-engineering-zoomcamp'},
             {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
              'section': 'General course-related questions',
              'question': 'Course - What can I do before the course starts?',
              'course': 'data-engineering-zoomcamp'},
             {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
              'section': 'General course-related questions',
              'question': 'How can we contribute to the course?',
              'course': 'data-engineering-zoomcamp'}
            ]

embeddings_vector = []

for document in documents:
    embeddings_doc = list(model.embed([document['text']]))
    embeddings_vector.append(embeddings_doc[0])

cos_sim = np.array(embeddings_vector).dot(embeddings_query[0])
np.argmin(cos_sim)
```
```Output
np.int64(3)
```
</details>

## Q4. Ranking by cosine, version two

What's the highest scoring document?

- 0
- 1
- 2
- 3
- 4
#### Solution :
<details open>
<summary> Expand</summary>
  
```cmd
embeddings_vector = []

for document in documents:
    full_text = document['question'] + ' ' + document['text']
    embeddings_doc = list(model.embed([full_text]))
    embeddings_vector.append(embeddings_doc[0])

cos_sim = np.array(embeddings_vector).dot(embeddings_query[0])
np.argmax(cos_sim)
```
```Output
np.int64(0)
```

Yes, both are different, Q3 compairing body of text while Q4 provides key information that improves the sementic similarity if query is similar to original question.

</details>

## Q5. Selecting the embedding model

What's the smallest dimensionality for models in fastembed?

- 128
- 256
- 384
- 512

#### Solution :
<details open>
<summary> Expand</summary>
  
```cmd
models_list = TextEmbedding.list_supported_models()

dimensions = []
for model in models_list:
    dimensions.append(model['dim'])
    
models_list[np.array(dimensions).argmin()]
```
```Output
{'model': 'BAAI/bge-small-en',
 'sources': {'hf': 'Qdrant/bge-small-en',
  'url': 'https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz',
  '_deprecated_tar_struct': True},
 'model_file': 'model_optimized.onnx',
 'description': 'Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2023 year.',
 'license': 'mit',
 'size_in_GB': 0.13,
 'additional_files': [],
 'dim': 384,
 'tasks': {}}
```
</details>


## Q6. Indexing with qdrant (2 points)

What's the highest score in the results?
(The score for the first returned record):

- 0.97
- 0.87
- 0.77
- 0.67
#### Solution :
<details open>
<summary> Expand</summary>
  
```cmd
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()


documents = []

for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

client = QdrantClient("http://localhost:6333")

EMBEDDING_DIMENSIONALITY = 384
collection_name = "ml-zoomcamp-rag"
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,
        distance=models.Distance.COSINE
    )
)

model_name = 'BAAI/bge-small-en'
points = []
id_ = 0
for doc in documents:
    text = doc['question'] + ' ' + doc['text']
    point = models.PointStruct(
        id=id_,
        vector=models.Document(text=text, model=model_name),
        payload={
            "text": text,
            "section": doc['section'],
            "course": doc['course']
        }
    )
    points.append(point)

    id_ += 1

client.upsert(
    collection_name=collection_name,
    points=points
)

def search(query, limit=1):

    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_name
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results

result = search(query).points[0]
result.score
```
```Output
0.8703172
```
</details>
