import unittest
import main

class TestMain(unittest.TestCase):

    def test_read_file(self):
        expected_output = ['127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326',
                           '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /favicon.ico HTTP/1.0" 404 208',
                           '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /favicon.ico HTTP/1.0" 404 208']
        
        output = main.read_file("test.log")
        
        self.assertEqual(output, expected_output)
        
    def test_parse_logs(self):
        input_logs = ['127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326',
                      '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /favicon.ico HTTP/1.0" 404 208',
                      '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /favicon.ico HTTP/1.0" 404 208']
        
        expected_output = {'/apache_pb.gif': {'requests': 1, 'status_codes': {'200': 1}, 'users': {'frank': 1}}, 
                           '/favicon.ico': {'requests': 2, 'status_codes': {'404': 2}, 'users': {'frank': 2}}}
        
        output = main.parse_logs(input_logs)
        
        self.assertEqual(output, expected_output)
        
if __name__ == '__main__':
    unittest.main()
