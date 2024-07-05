class Item:
    def __init__(self, name, description, p_s):
        self.name = name
        self.description = description
        self.p_s = p_s

    def update_p_s(self, p_s):
        self.p_s = p_s

    def get_name(self):
        return self.name
    
    def get_item_info(self) -> None:
        return f"name={self.name}, p_s={self.p_s}"
