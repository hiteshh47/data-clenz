pipeline {
    agent any

    stages {
        stage('Build Docker image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKERHUB_PASSWD')]) {
                    //sh "docker rmi -f hiteshdev47/data-clenz-app:latest"
                    sh "docker login -u $DOCKER_USERNAME -p $DOCKERHUB_PASSWD"
                    sh 'docker build -t hiteshdev47/data-clenz-app:latest .'
                    sh 'docker push hiteshdev47/data-clenz-app:latest'
                }
            }
        }
        stage('Test') {
            steps {
                script{
                    sshPublisher(publishers: [sshPublisherDesc(configName: 'data-clenz', transfers: [sshTransfer(cleanRemote: false,
                    execCommand: 'sh dock.sh', execTimeout: 120000, flatten: false, makeEmptyDirs: false, 
                    noDefaultExcludes: false, patternSeparator: '[, ]+', remoteDirectory: '',
                    remoteDirectorySDF: false, removePrefix: '', sourceFiles: '')],
                    usePromotionTimestamp: false, useWorkspaceInPromotion: false, verbose: false)])
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
