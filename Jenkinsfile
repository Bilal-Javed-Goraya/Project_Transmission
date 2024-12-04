pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/Bilal-Javed-Goraya/Project_Transmission.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t your-image-name:latest .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh '''
                docker ps -q --filter "name=streamlit_app" | grep -q . && docker stop streamlit_app && docker rm streamlit_app || true
                docker run -d --name streamlit_app -p 8501:8501 your-image-name:latest
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline completed!'
        }
    }
}
