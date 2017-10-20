a = {0: 5, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 43, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 'turn': 'player_2'}
index = 100
print ''.join(str([index]+[v for k,v in a.iteritems()]).split()).strip('[]')