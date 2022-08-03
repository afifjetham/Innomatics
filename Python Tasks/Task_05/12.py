from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        prefix = '\n' in data and 'Multi-line Comment' or 'Single-line Comment'
        print('>>> {0}\n{1}'.format(prefix, data))
    def handle_data(self, data):
        if data is not '\n':
            print('>>> Data\n{0}'.format(data))
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()