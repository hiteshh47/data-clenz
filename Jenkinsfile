pipeline {
    agent any

    stages {
        stage('Build Docker image') {
            steps {
                sh 'docker login -u "hiteshdev47" -p "Hite$h47docker"'
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
