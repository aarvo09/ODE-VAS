import re

def remove_comments_from_python(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        result = []
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.lstrip()
            
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        continue
                    else:
                        in_docstring = True
                        continue
                elif stripped.startswith('#'):
                    continue
                else:
                    result.append(line)
            else:
                if docstring_char in line:
                    in_docstring = False
                continue
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result))
        return True
    except:
        return False

if __name__ == '__main__':
    remove_comments_from_python('app.py')
