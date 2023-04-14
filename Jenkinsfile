pipeline {
    agent any

    stages {
        stage('Build Docker image') {
            steps {
                sh 'docker login -u ${env.DOCKER_USERNAME} -p ${DOCKERHUB_PASSWD}'
                sh 'docker build -t hiteshdev47/data-clenz-app:latest .'
                sh 'docker push hiteshdev47/data-clenz-app:latest'
            }
        }
        stage('Test') {
            steps {
                script{
                    sshPublisher(publishers: [sshPublisherDesc(configName: 'data-clenz', transfers: [sshTransfer(cleanRemote: false, excludes: '', execCommand: '''sudo docker rm -f data-clenz
                    sudo docker rmi hiteshdev47/data-clenz-app
                    sudo docker login -u hiteshdev47 -p Hitesh47docker
                    sudo docker pull hiteshdev47/data-clenz-app:latest
                    sudo docker run -d --name data-clenz -p 80:8501 hiteshdev47/data-clenz-app
                    sudo docker ps''', execTimeout: 120000, flatten: false, makeEmptyDirs: false, noDefaultExcludes: false, patternSeparator: '[, ]+', remoteDirectory: '', remoteDirectorySDF: false, removePrefix: '', sourceFiles: '')], usePromotionTimestamp: false, useWorkspaceInPromotion: false, verbose: false)])
                }
            }   
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
