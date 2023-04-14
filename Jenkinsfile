pipeline {
    agent any

    stages {
        stage('Build Docker image') {
            steps {
                sh 'docker build -t data-clenz-app'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
