pipeline {
    agent any

    stages {
        stage('Build Docker image') {
            steps {
                sh 'docker login -u "hiteshdev47" -p "Hitesh47docker"'
                sh 'docker build -t hiteshdev47/data-clenz-app:latest .'
                sh 'docker push hiteshdev47/data-clenz-app:latest'
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
