# MindBlown Chatbot

## Project Description
The Recommendation System is designed to provide activity recommendation that match the users of the MindBlown application, using the concepts of Content-Based Filtering and Collaborative Filtering to generate the best possible recommendations. Content-Based Filtering uses the concept of cosine similarity to find similarities between activities, while Collaborative Filtering uses the RecommenderNet architecture. Flask is used to create the appropriate endpoints for deployment, and Google Cloud Run is used to deploy the system. 

## Features
- Content-Based Filtering for recommending activities based on similar activities
- Collaborative Filtering for recommending activities based on other users' interactions
- Fast and lightweight deployment using TensorFlow Lite

---

## Running Locally

Follow the steps below to run the recommendation system on your local machine:

### 1. Clone the Repository
```
git clone https://github.com/MindBlownDBS/Recomendation-System-Model.git
cd Recomendation-System-Model
```

### 2. Build and Run Docker
```
docker build -t sistem-rekomendasi .
docker run -p 5000:5000 sistem-rekomendasi
```

## Deploy in Google Cloud Run

Follow the steps below to deploy your model in Google Cloud Run:

### 1. Clone the Repository
```
git clone https://github.com/MindBlownDBS/Recomendation-System-Model.git
cd Recomendation-System-Model
```

### 2. Build Docker
```
docker build -t sistem-rekomendasi .
```

### 3. Create Artifact Registry Repository
```
gcloud artifacts repositories create NAMA_REPOSITORY \
    --repository-format=docker \
    --location=REGION \
    --description="Deskripsi repository Anda"
```

### 4. Tag and Push Docker Image to Artifact Registry
```
docker tag nama-image-anda:tag-lokal REGION-docker.pkg.dev/PROJECT_ID/NAMA_REPOSITORY/nama-image-anda:tag-remote

docker push REGION-docker.pkg.dev/PROJECT_ID/NAMA_REPOSITORY/nama-image-anda:tag-remote
```

### 5. Deploy Image to Cloud Run
```
gcloud run deploy NAMA-LAYANAN-CLOUDRUN \
    --image=REGION-docker.pkg.dev/PROJECT_ID/NAMA_REPOSITORY/nama-image-anda:tag-remote \
    --platform=managed \
    --region=REGION \
    --allow-unauthenticated \
    --port=PORT_DALAM_DOCKERFILE \
    --memory=ALOKASI_MEMORI \
    --cpu=ALOKASI_CPU \
    --min-instances=0 \
    --max-instances=MAKSIMUM_INSTANCES
```

Once everything is uploaded and secrets are configured, Hugging Face will automatically build and run your container.
