pipeline {
    agent any
    environment{
        DOCKERHUB = credentials('dockerhub')
    }

    stages {
        stage('Build Docker image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKERHUB_PASSWD')]) {
                    //sh "docker rmi -f hiteshdev47/data-clenz-app:latest"
                    sh "echo $DOCKERHUB_PASSWD | docker login -u $DOCKER_USERNAME --password-stdin "
                    sh 'docker build -t hiteshdev47/data-clenz-app:${BUILD_NUMBER} .'
                    sh 'docker push hiteshdev47/data-clenz-app:${BUILD_NUMBER}'
                }
            }
        }
        stage('Test') {
            steps {
                script{
                    sshPublisher(publishers: [sshPublisherDesc(configName: 'data-clenz', transfers: [sshTransfer(cleanRemote: false,
                    execCommand: 'echo $DOCKERHUB_PSW |docker login -u $DOCKERHUB_USR --password-stdin; 
                    docker pull hiteshdev47/data-clenz-app:${BUILD_NUMBER};
                    sh dock.sh ', 
                    execTimeout: 120000, flatten: false, makeEmptyDirs: false, 
                    noDefaultExcludes: false, patternSeparator: '[, ]+', remoteDirectory: '',
                    remoteDirectorySDF: false, removePrefix: '', sourceFiles: '')],
                    usePromotionTimestamp: false, useWorkspaceInPromotion: false, verbose: false)])
                }
            }   
        }
        stage('Trigger another Job') {
            steps {
               build job: 'downstreamJob', parameters: [string(name: 'build_params', value: env.BUILD_NUMBER)]
            }
        }
    }
}
