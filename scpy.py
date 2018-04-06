import os
import pexpect
from conf import conf


class DirectoriesSync(object):
    def __init__(self, local, remote):
        self.localDir = os.path.abspath(local)
        self.remoteSite = remote

    def push_remote_site(self):
        r = '-r'
        self.scp(r + ' ' + self.localDir + ' ' + self.remoteSite['user'] + '@' + self.remoteSite['host'] + ':' + self.remoteSite['dest'], self.remoteSite['creds'])

    def scp(self, cmd, password=None):
        print('SCPing... scp ' + cmd)
        child = pexpect.spawn('scp ' + cmd, timeout=300)

        expect = ['The authenticity of host', 'Permission denied', 'password', 'refused', pexpect.EOF, pexpect.TIMEOUT]
        index = child.expect(expect)
        if index == 0:
            child.sendline('yes')
        elif index == 1:
            child.sendcontrol('c')
        elif index == 2:
            child.sendline(password)
        elif index == 3:
            print('Connection refused...')
        elif index == 4:
            print('Unexpected end of line')
        elif index == 5:
            print('Timed out...')
            raise Exception
        index = child.expect(['denied', pexpect.EOF])
        if index == 0:
            child.sendcontrol('c')
            print('Permission denied')
            print('Command failed:', 'scp ' + cmd)
        else:
            print(child.before)

def sync_model(model_name=conf['BEST_MODEL']):
    #  Sync trained model from Training Server to Self-play Server
    remote_server = conf['SELF_PLAY_SERVER']
    remote_server['dest'] = conf['MODEL_DIR']
    local_model_file = os.path.join(conf['MODEL_DIR'], model_name)
    print('Syncing model %s to remote place %s:%s', local_model_file, remote_server['host'], remote_server['dest'])
    d = DirectoriesSync(local_model_file, remote_server)
    d.push_remote_site()

def sync_game_data(game_dir, model_name, game_name):
    #  Sync game from Self-play Server to Training Server
    remote_server = conf['TRAINING_SERVER']
    remote_server['dest'] = os.path.join(game_dir, model_name)
    local_dir = os.path.join(game_dir, model_name, game_name)
    print('Syncing game data %s to remote place %s', local_dir, remote_server['host'], remote_server['dest'])
    d = DirectoriesSync(local_dir, remote_server)
    d.push_remote_site()

def sync_all_game_data(game_dir, model_name):
    for d in os.listdir(os.path.join(game_dir, model_name)):
        sync_game_data(game_dir, model_name, d)

def main():
    locals_dir = 'sp_models/best_model.h5'
    remotes = {
        'user': 'miruware',
        'host': '211.180.114.9',
        'dest': '/home/miruware/temp',
        'creds': 'miruware@'
    }
    d = DirectoriesSync(locals_dir, remotes)
    d.push_remote_site()

if __name__ == "__main__":
    sync_game_data(conf['SELF_PLAY_DIR'], "model_1")
    main()
