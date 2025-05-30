from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# --- Variabel Global untuk Content Based Filtering ---
activities_df = None
cosine_sim_matrix = None
activity_indices_series = None

# --- Variabel Global untuk Collaborative Filtering ---
user_to_user_encoded_map = None
user_encoded_to_user_map = None
activity_to_activity_encoded_map = None
activity_encoded_to_activity_map = None
ratings_df_global = None
all_activities_cf_encoded_list = None

def load_and_preprocess_data():
    """
    Memuat data aktivitas dan melakukan pra-pemrosesan untuk Content Based Filtering
    dan Collaborative Filtering. Fungsi ini dijalankan sekali saat aplikasi Flask dimulai.
    """
    global activities_df, cosine_sim_matrix, activity_indices_series
    global user_to_user_encoded_map, user_encoded_to_user_map
    global activity_to_activity_encoded_map, activity_encoded_to_activity_map
    global ratings_df_global, all_activities_cf_encoded_list
    
    # === BAGIAN CONTENT BASED FILTERING ===
    try:
        activities_df = pd.read_csv('dataset/activity.csv')
        activities_df['features'] = activities_df['category'].astype(str) + ' ' + \
                                    activities_df['duration'].astype(str) + ' ' + \
                                    activities_df['energy_needed'].astype(str)
        tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix_transformed = tfidf_vectorizer.fit_transform(activities_df['features'])
        cosine_sim_matrix = linear_kernel(tfidf_matrix_transformed, tfidf_matrix_transformed)
        activity_indices_series = pd.Series(activities_df.index, index=activities_df['name']).drop_duplicates()
        print("Data untuk Content Based Filtering berhasil dimuat dan diproses.")
    except FileNotFoundError:
        print("Error CBF: File 'activity.csv' tidak ditemukan.")
        activities_df = pd.DataFrame(columns=['activityId', 'name', 'category', 'duration', 'energy_needed'])
        cosine_sim_matrix = np.array([])
        activity_indices_series = pd.Series(dtype='float64')
    except Exception as e:
        print(f"Error CBF saat memuat/memproses data: {e}")
        activities_df = pd.DataFrame(columns=['activityId', 'name', 'category', 'duration', 'energy_needed'])
        cosine_sim_matrix = np.array([])
        activity_indices_series = pd.Series(dtype='float64')

    # === BAGIAN COLLABORATIVE FILTERING ===
    try:
        # Pastikan file 'rating.csv' berada di direktori yang sama atau pathnya benar
        ratings_df_global = pd.read_csv('dataset/rating.csv')
        
        import json

        # Load mapping user & activity
        with open("data/user_to_user_encoded.json") as f:
            user_to_user_encoded_map = json.load(f)

        with open("data/user_encoded_to_user.json", "r") as f:
            user_encoded_to_user_map = json.load(f)

        with open("data/activity_to_activity_encoded.json", "r") as f:
            activity_to_activity_encoded_map = json.load(f)

        with open("data/activity_encoded_to_activity.json", "r") as f:
            activity_encoded_to_activity_map = json.load(f)

        # Konversi key json dari string ke int (opsional tapi aman)
        user_to_user_encoded_map = {k: v for k, v in user_to_user_encoded_map.items()}
        user_encoded_to_user_map = {int(k): v for k, v in user_encoded_to_user_map.items()}
        activity_to_activity_encoded_map = {k: v for k, v in activity_to_activity_encoded_map.items()}
        activity_encoded_to_activity_map = {int(k): v for k, v in activity_encoded_to_activity_map.items()}
        
        # Simpan list semua ID aktivitas ter-encode yang diketahui model
        all_activities_cf_encoded_list = list(activity_to_activity_encoded_map.values())

        print("Data pendukung CF berhasil dimuat.")

    except FileNotFoundError as e:
        print(f"Error CF: File tidak ditemukan - {e}. Collaborative Filtering mungkin tidak berfungsi.")
        ratings_df_global = pd.DataFrame()
    except Exception as e:
        print(f"Error CF saat memuat data: {e}")
        ratings_df_global = pd.DataFrame()

# Panggil fungsi untuk memuat dan memproses data saat aplikasi pertama kali dijalankan
load_and_preprocess_data()

@app.route('/recommend/content-based', methods=['GET'])
def recommend_content_based_api():
    activity_name_input = request.args.get('activity_name')
    top_n_input_str = request.args.get('top_n', default='10')

    if not activity_name_input:
        return jsonify({"error": "Parameter 'activity_name' tidak boleh kosong."}), 400
    try:
        top_n = int(top_n_input_str)
        if top_n <= 0:
            return jsonify({"error": "Parameter 'top_n' harus bilangan bulat positif."}), 400
    except ValueError:
        return jsonify({"error": "Parameter 'top_n' harus berupa bilangan bulat."}), 400

    if activities_df is None or activities_df.empty or cosine_sim_matrix is None or cosine_sim_matrix.size == 0 or activity_indices_series is None or activity_indices_series.empty:
        return jsonify({"error": "Data aktivitas CBF tidak termuat dengan benar di server."}), 500
    
    activity_idx = activity_indices_series.get(activity_name_input)
    if activity_idx is None:
        return jsonify({"error": f"Aktivitas '{activity_name_input}' tidak ditemukan (CBF)."}), 404
    if not isinstance(activity_idx, (int, np.integer)):
        if isinstance(activity_idx, pd.Series) and not activity_idx.empty:
            activity_idx = int(activity_idx.iloc[0])
        else:
            return jsonify({"error": f"Indeks untuk aktivitas '{activity_name_input}' tidak valid (CBF)."}), 500
    else:
        activity_idx = int(activity_idx)

    try:
        sim_scores = list(enumerate(cosine_sim_matrix[activity_idx]))
    except IndexError:
        return jsonify({"error": f"Indeks {activity_idx} di luar jangkauan untuk similarity matrix (CBF)."}), 500
    
    sim_scores = [score for score in sim_scores if score[0] != activity_idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    recommended_indices = [i[0] for i in sim_scores]
    recommendations_data = activities_df.iloc[recommended_indices][['name', 'category', 'duration', 'energy_needed']]
    recommendations_output = []
    for i, score_tuple in enumerate(sim_scores):
        rec_data = recommendations_data.iloc[i].to_dict()
        rec_data['similarity_score'] = round(score_tuple[1], 4)
        recommendations_output.append(rec_data)
    return jsonify({
        "recommendations_for": activity_name_input,
        "type": "Content-Based Filtering",
        "top_recommendations": recommendations_output
    })

@app.route('/recommend/collaborative-tflite', methods=['GET'])
def recommend_collaborative_tflite_api():
    """
    Endpoint untuk rekomendasi aktivitas menggunakan model TFLite.
    Parameter:
        user_id (str): ID pengguna seperti 'User46'
        top_n (int): jumlah rekomendasi teratas
    """
    user_id_input = request.args.get('user_id')
    top_n_input_str = request.args.get('top_n', default='10')

    if not user_id_input:
        return jsonify({"error": "Parameter 'user_id' tidak boleh kosong."}), 400

    try:
        top_n = int(top_n_input_str)
        if top_n <= 0:
            return jsonify({"error": "Parameter 'top_n' harus bilangan bulat positif."}), 400
    except ValueError:
        return jsonify({"error": "Parameter 'top_n' harus berupa bilangan bulat."}), 400

    # Pastikan semua data sudah dimuat
    if ratings_df_global is None or ratings_df_global.empty or \
       user_to_user_encoded_map is None or activity_to_activity_encoded_map is None or \
       activity_encoded_to_activity_map is None or activities_df is None or activities_df.empty:
        return jsonify({"error": "Data pendukung Collaborative Filtering belum dimuat sepenuhnya."}), 500

    # Load model TFLite interpreter (hanya sekali per request)
    try:
        interpreter = tflite.Interpreter(model_path="models/recommender_model.tflite")
        interpreter.allocate_tensors()
    except Exception as e:
        return jsonify({"error": f"Gagal memuat model TFLite: {e}"}), 500

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    user_encoder_val = user_to_user_encoded_map.get(user_id_input)
    if user_encoder_val is None:
        return jsonify({"error": f"User ID '{user_id_input}' tidak ditemukan dalam data training."}), 404

    # Data aktivitas yang belum ditonton user
    watched_activity_ids = ratings_df_global[ratings_df_global['userId'] == user_id_input]['activityId'].values
    all_known_activity_ids = list(activity_to_activity_encoded_map.keys())
    not_watched_original_ids = np.setdiff1d(all_known_activity_ids, watched_activity_ids)

    if len(not_watched_original_ids) == 0:
        return jsonify({
            "message": f"Tidak ada aktivitas baru yang bisa direkomendasikan untuk user {user_id_input}.",
            "recommendations_for_user_id": user_id_input,
            "type": "Collaborative Filtering (TFLite)",
            "top_recommendations": []
        }), 200

    activity_not_watched_encoded = [
        [activity_to_activity_encoded_map[x]] for x in not_watched_original_ids
        if x in activity_to_activity_encoded_map
    ]

    if not activity_not_watched_encoded:
        return jsonify({
            "message": f"Tidak ada aktivitas valid untuk diproses oleh model.",
            "recommendations_for_user_id": user_id_input,
            "type": "Collaborative Filtering (TFLite)",
            "top_recommendations": []
        }), 200

    # Buat input [user_id_encoded, activity_id_encoded]
    user_activity_array = np.hstack(
        ([[user_encoder_val]] * len(activity_not_watched_encoded), np.array(activity_not_watched_encoded))
    ).astype(np.float32)

    predictions = []
    for i in range(user_activity_array.shape[0]):
        input_data = user_activity_array[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
        predictions.append((i, pred_score))

    top_n = min(top_n, len(predictions))
    top_indices = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommended_activity_ids_encoded = [activity_not_watched_encoded[i[0]][0] for i in top_indices]
    recommended_activity_ids_original = [activity_encoded_to_activity_map.get(i) for i in recommended_activity_ids_encoded]

    recs_detail = activities_df[activities_df['activityId'].isin(recommended_activity_ids_original)].copy()
    temp_scores_df = pd.DataFrame({
        'activityId': recommended_activity_ids_original,
        'predicted_score': [round(i[1], 4) for i in top_indices]
    })

    final_df = pd.merge(temp_scores_df, recs_detail, on='activityId', how='left').sort_values(by='predicted_score', ascending=False)
    output = final_df[['name', 'category', 'duration', 'energy_needed', 'predicted_score']].to_dict(orient='records')

    return jsonify({
        "recommendations_for_user_id": user_id_input,
        "type": "Collaborative Filtering (TFLite)",
        "top_recommendations": output
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)