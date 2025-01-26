from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import json
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pandas as pd
from urllib.parse import quote_plus as url_quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Database Configuration
DB_DIR = 'database'
USERS_FILE = os.path.join(DB_DIR, 'users.json')
POSTS_FILE = os.path.join(DB_DIR, 'posts.json')
GROUPS_FILE = os.path.join(DB_DIR, 'groups.json')
INTERESTS_FILE = os.path.join(DB_DIR, 'interests.json')
INTERACTIONS_FILE = os.path.join(DB_DIR, 'interactions.json')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'mp4', 'mov', 'avi','webm'}  # Add video formats
# Ensure database directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Database Initialization Functions
def init_json_file(filepath, default_data):
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_data, f)
        print(f"Initialized {filepath}")

# Initialize Files
init_json_file(USERS_FILE, {'users': []})
init_json_file(POSTS_FILE, {'posts': [
    {
        'id': '1',
        'user_id': 'system',
        'content': 'Welcome to the platform! Create your first post.',
        'created_at': datetime.now().isoformat(),
        'likes': 0,
        'comments': []
    }
]})
init_json_file(GROUPS_FILE, {'groups': []})
init_json_file(INTERESTS_FILE, {'interests': ['java', 'python', 'ruby', 'javascript', 'cplusplus', 'go', 'rust', 'swift', 'objective-c', 'assembly', 'shell', 'perl', 'php']})
init_json_file(INTERACTIONS_FILE, {'interactions': []})

# Database Operations
def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'posts': [], 'users': [], 'interactions': []}
    


def save_data(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
'''
# Advanced Recommendation System
def advanced_recommendation(user_id):
    users_data = load_data(USERS_FILE)
    posts_data = load_data(POSTS_FILE)
    
    # Collaborative Filtering
    def calculate_similarity(user1, user2):
        common_interests = set(user1['interests']) & set(user2['interests'])
        return len(common_interests) / len(set(user1['interests'] + user2['interests']))
    
    # Content-Based Filtering
    def content_based_recommendation(posts):
        vectorizer = TfidfVectorizer(stop_words='english')
        post_contents = [post['content'] for post in posts]
        
        if not post_contents:
            return []
        
        tfidf_matrix = vectorizer.fit_transform(post_contents)
        similarity_scores = cosine_similarity(tfidf_matrix)
        
        recommended_posts = []
        for i, post in enumerate(posts):
            avg_similarity = similarity_scores[i].mean()
            post['similarity_score'] = avg_similarity
            recommended_posts.append(post)
        
        return sorted(recommended_posts, key=lambda x: x['similarity_score'], reverse=True)[:10]
    
    # Find similar users
    current_user = next((u for u in users_data['users'] if u['id'] == user_id), None)
    if not current_user:
        return posts_data['posts'][:10], []
    
    similar_users = [
        user for user in users_data['users'] 
        if user['id'] != user_id and calculate_similarity(current_user, user) > 0.3
    ]
    
    # Recommend posts
    recommended_posts = []
    for post in posts_data['posts']:
        post_author = next((u for u in users_data['users'] if u['id'] == post['user_id']), None)
        if post_author in similar_users:
            recommended_posts.append(post)
    
    # Combine collaborative and content-based recommendations
    content_based_recs = content_based_recommendation(posts_data['posts'])
    
    # Merge and deduplicate recommendations
    final_recommendations = list(dict.fromkeys(recommended_posts + content_based_recs))[:10]
    
    # Fallback to all posts if no recommendations
    return final_recommendations or posts_data['posts'][:10], similar_users[:5]
'''
recommendations = []
recommendations.sort(key=lambda x: x[1], reverse=True)
recommended_posts = [post for post, _ in recommendations[:10]]
# Collaborative Filtering Algorithm using SVD
def get_recommendations(user_id):
    users_data = load_data(USERS_FILE)
    posts_data = load_data(POSTS_FILE)

    # Prepare data for Surprise
    ratings = []
    for post in posts_data['posts']:
        ratings.append((post['user_id'], post['id'], 1))  # Assuming each post is rated as 1

    df = pd.DataFrame(ratings, columns=['user_id', 'post_id', 'rating'])
    reader = Reader(rating_scale=(1, 1))
    data = Dataset.load_from_df(df[['user_id', 'post_id', 'rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Use SVD for matrix factorization
    model = SVD()
    model.fit(trainset)

    # Get top N recommendations for the user
    user_posts = df[df['user_id'] == user_id]['post_id'].tolist()
    all_posts = df['post_id'].unique()
    recommendations = []

    for post in all_posts:
        if post not in user_posts:
            pred = model.predict(user_id, post)
            recommendations.append((post, pred.est))

    # Sort recommendations by estimated rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_posts = [post for post, _ in recommendations[:10]]

    # Get recommended users (for simplicity, we can use the same logic)
    recommended_users = [user for user in users_data['users'] if user['id'] != user_id][:5]

    return recommended_posts, recommended_users

def advanced_recommendation(user_id):
    users_data = load_data(USERS_FILE)
    posts_data = load_data(POSTS_FILE)

    # Find current user
    current_user = next((u for u in users_data['users'] if u['id'] == user_id), None)
    if not current_user:
        return posts_data['posts'][:10], []

    # Collaborative Filtering
    def calculate_similarity(user1, user2):
        common_interests = set(user1.get('interests', [])) & set(user2.get('interests', []))
        total_interests = set(user1.get('interests', [])) | set(user2.get('interests', []))
        return len(common_interests) / len(total_interests) if total_interests else 0

    similar_users = [
        user for user in users_data['users'] 
        if user['id'] != user_id and calculate_similarity(current_user, user) > 0.3
    ]

    # Content-Based Filtering
    def content_based_recommendation(posts):
        vectorizer = TfidfVectorizer(stop_words='english')
        post_contents = [post.get('content', '') for post in posts]
        
        if not post_contents:
            return posts[:10]
        
        tfidf_matrix = vectorizer.fit_transform(post_contents)
        similarity_scores = cosine_similarity(tfidf_matrix)
        
        recommended_posts = []
        for i, post in enumerate(posts):
            post['similarity_score'] = similarity_scores[i].mean()
            recommended_posts.append(post)
        
        return sorted(recommended_posts, key=lambda x: x.get('similarity_score', 0), reverse=True)[:10]

    # Get recommendations
    content_based_recs = content_based_recommendation(posts_data['posts'])

    # Deduplicate recommendations
    seen_post_ids = set()
    final_recommendations = []
    for post in content_based_recs:
        if post['id'] not in seen_post_ids:
            seen_post_ids.add(post['id'])
            final_recommendations.append(post)
    print("Current User Interests:", current_user.get('interests', []))
    print("Similar Users Found:", similar_users)
    print("Total posts available:", len(posts_data['posts']))
    print("Recommended posts:", recommended_posts)

    return final_recommendations, similar_users[:5]
# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function





# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('is_admin'):
            flash('Admin access required')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function



# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users_data = load_data(USERS_FILE)
        interests_data = load_data(INTERESTS_FILE)
        
        email = request.form['email']
        password = request.form['password']
        interests = request.form.getlist('interests')
        
        # Check if user exists
        if any(user['email'] == email for user in users_data['users']):
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = {
            'id': str(len(users_data['users']) + 1),
            'email': email,
            'password': generate_password_hash(password),
            'interests': interests,
            'created_at': datetime.now().isoformat(),
            'is_admin': False
        }
        
        users_data['users'].append(new_user)
        save_data(USERS_FILE, users_data)
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    interests = load_data(INTERESTS_FILE)['interests']
    return render_template('register.html', interests=interests)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users_data = load_data(USERS_FILE)
        email = request.form['email']
        password = request.form['password']
        
        user = next((u for u in users_data['users'] if u['email'] == email), None)
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['is_admin'] = user.get('is_admin', False)
            return redirect(url_for('dashboard'))
            
        flash('Invalid credentials')
    return render_template('login.html')


# Dashboard Route with Enhanced Recommendations
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get recommendations
        recommended_posts, recommended_users = advanced_recommendation(session['user_id'])
        
        # Ensure posts exist
        if not recommended_posts:
            posts_data = load_data(POSTS_FILE)
            recommended_posts = posts_data['posts']
        
        # Load additional data
        groups_data = load_data(GROUPS_FILE)
        
        return render_template('dashboard.html',
                               posts=recommended_posts,
                               recommended_users=recommended_users,
                               groups=groups_data['groups'])
    
    except Exception as e:
        # Error handling
        print(f"Recommendation Error: {e}")
        posts_data = load_data(POSTS_FILE)
        return render_template('dashboard.html', 
                               posts=posts_data['posts'], 
                               error="Recommendation failed")



@app.route('/profile')
@login_required
def profile():
    users_data = load_data(USERS_FILE)
    user = next((u for u in users_data['users'] if u['id'] == session['user_id']), None)
    return render_template('profile.html', user=user)

'''
@app.route('/create_post', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        posts_data = load_data(POSTS_FILE)
        
        # Handle text content
        content = request.form['content']
        description = request.form['description']
        tags = request.form.getlist('tags')  # Assuming tags are sent as a list

        # Handle file uploads
        media_files = []
        for file in request.files.getlist('media_files'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                media_files.append(filepath)

        new_post = {
            'id': str(len(posts_data['posts']) + 1),
            'user_id': session['user_id'],
            'content': content,
            'description': description,
            'tags': tags,
            'media_files': media_files,  # Store paths for images/videos
            'created_at': datetime.now().isoformat(),
            'likes': 0,
            'comments': []
        }
        
        posts_data['posts'].append(new_post)
        save_data(POSTS_FILE, posts_data)
        
        return redirect(url_for('dashboard'))
    
    return render_template('create_post.html')

'''

@app.route('/create_post', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        posts_data = load_data(POSTS_FILE)
        
        # Handle file uploads
        media_files = []
        for file in request.files.getlist('media_files'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # This should be correct
                file.save(filepath)
                media_files.append(filename)  # Store just the filename

        new_post = {
            'id': str(len(posts_data['posts']) + 1),
            'user_id': session['user_id'],
            'content': request.form['content'],
            'description': request.form.get('description', ''),
            'tags': request.form.getlist('tags'),
            'media_files': media_files,  # Store filenames
            'created_at': datetime.now().isoformat(),
            'likes': 0,
            'comments': []
        }
        
        posts_data['posts'].append(new_post)
        save_data(POSTS_FILE, posts_data)
        
        return redirect(url_for('dashboard'))
    
    return render_template('create_post.html')
@app.route('/groups')
@login_required
def groups():
    groups_data = load_data(GROUPS_FILE)
    return render_template('groups.html', groups=groups_data['groups'])

@app.route('/create_group', methods=['GET', 'POST'])
@login_required
def create_group():
    if request.method == 'POST':
        groups_data = load_data(GROUPS_FILE)
        
        new_group = {
            'id': str(len(groups_data['groups']) + 1),
            'name': request.form['name'],
            'description': request.form['description'],
            'creator_id': session['user_id'],
            'members': [session['user_id']],
            'created_at': datetime.now().isoformat()
        }
        
        groups_data['groups'].append(new_group)
        save_data(GROUPS_FILE, groups_data)
        
        return redirect(url_for('groups'))
    
    return render_template('create_group.html')

@app.route('/admin')
@admin_required
def admin_dashboard():
    users_data = load_data(USERS_FILE)
    posts_data = load_data(POSTS_FILE)
    groups_data = load_data(GROUPS_FILE)
    
    return render_template('admin.html',
                         users=users_data['users'],
                         posts=posts_data['posts'],
                         groups=groups_data['groups'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# API endpoints for AJAX operations
@app.route('/api/like_post/<post_id>', methods=['POST'])
@login_required
def like_post(post_id):
    posts_data = load_data(POSTS_FILE)  # Load the posts data
    post = next((p for p in posts_data['posts'] if p['id'] == post_id), None)  # Find the post by ID
    
    if post:
        # Check if the user has already liked the post
        if 'liked_by' not in post:
            post['liked_by'] = []  # Initialize liked_by if it doesn't exist
        
        if session['user_id'] in post['liked_by']:
            return jsonify({'success': False, 'message': 'You have already liked this post.'}), 400  # User already liked the post
        
        # Increment the like count and add user ID to liked_by
        post['likes'] += 1
        post['liked_by'].append(session['user_id'])  # Add user ID to liked_by
        save_data(POSTS_FILE, posts_data)  # Save the updated posts data
        return jsonify({'success': True, 'likes': post['likes']})  # Return the updated like count
    
    return jsonify({'success': False}), 404  # Return error if post not found


@app.route('/api/join_group/<group_id>', methods=['POST'])
@login_required
def join_group(group_id):
    groups_data = load_data(GROUPS_FILE)
    group = next((g for g in groups_data['groups'] if g['id'] == group_id), None)
    
    if group and session['user_id'] not in group['members']:
        group['members'].append(session['user_id'])
        save_data(GROUPS_FILE, groups_data)
        return jsonify({'success': True})
    
    return jsonify({'success': False}), 404

# Add these imports to the top of app.py
from werkzeug.utils import secure_filename

# Add these configurations
UPLOAD_FOLDER = 'static/uploads' 
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'mp4', 'mov', 'avi', 'webm', 'key'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add new JSON file for messages
MESSAGES_FILE = os.path.join(DB_DIR, 'messages.json')
init_json_file(MESSAGES_FILE, {'messages': []})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create admin user if it doesn't exist
users_data = load_data(USERS_FILE)
if not any(user.get('is_admin') for user in users_data['users']):
    admin_user = {
        'id': 'admin',
        'email': 'admin@admin.com',
        'password': generate_password_hash('admin123'),  # Change this in production!
        'interests': ['Administration'],
        'created_at': datetime.now().isoformat(),
        'is_admin': True
    }
    users_data['users'].append(admin_user)
    save_data(USERS_FILE, users_data)

@app.route('/group/<group_id>')
@login_required
def group_chat(group_id):
    groups_data = load_data(GROUPS_FILE)
    messages_data = load_data(MESSAGES_FILE)
    
    group = next((g for g in groups_data['groups'] if g['id'] == group_id), None)
    if not group:
        flash('Group not found')
        return redirect(url_for('groups'))
        
    if session['user_id'] not in group['members']:
        flash('You are not a member of this group')
        return redirect(url_for('groups'))
        
    group_messages = [m for m in messages_data['messages'] if m['group_id'] == group_id]
    users_data = load_data(USERS_FILE)
    
    return render_template('group_chat.html', 
                         group=group, 
                         messages=group_messages,
                         users=users_data['users'])

@app.route('/api/send_message', methods=['POST'])
@login_required
def send_message():
    messages_data = load_data(MESSAGES_FILE)
    
    message_text = request.form.get('message')
    group_id = request.form.get('group_id')
    file = request.files.get('file')
    
    message = {
        'id': str(len(messages_data['messages']) + 1),
        'user_id': session['user_id'],
        'group_id': group_id,
        'content': message_text,
        'created_at': datetime.now().isoformat(),
        'file_path': None
    }
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        message['file_path'] = filename
    
    messages_data['messages'].append(message)
    save_data(MESSAGES_FILE, messages_data)
    
    return jsonify({'success': True, 'message': message})


@app.route('/my_posts')
@login_required
def my_posts():
    users_data = load_data(USERS_FILE)
    posts_data = load_data(POSTS_FILE)

    # Get the current user's posts
    user_posts = [post for post in posts_data['posts'] if post['user_id'] == session['user_id']]

    return render_template('my_posts.html', posts=user_posts)


@app.route('/edit_post/<post_id>', methods=['GET', 'POST'])
@login_required
def edit_post(post_id):
    posts_data = load_data(POSTS_FILE)
    post = next((p for p in posts_data['posts'] if p['id'] == post_id), None)

    if request.method == 'POST':
        # Update the post content
        post['content'] = request.form['content']
        post['description'] = request.form.get('description', '')
        save_data(POSTS_FILE, posts_data)
        flash('Post updated successfully!')
        return redirect(url_for('my_posts'))

    return render_template('edit_post.html', post=post)

@app.route('/api/delete_post/<post_id>', methods=['DELETE'])
@login_required
def delete_post(post_id):
    posts_data = load_data(POSTS_FILE)
    post = next((p for p in posts_data['posts'] if p['id'] == post_id), None)

    if post and post['user_id'] == session['user_id']:
        posts_data['posts'].remove(post)
        save_data(POSTS_FILE, posts_data)
        return jsonify({'success': True}), 200

    return jsonify({'success': False}), 404
if __name__ == '__main__':
    app.run(debug=True)