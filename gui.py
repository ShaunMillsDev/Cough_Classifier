from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk

# Imports to allow to call any functions from main python file, with GUI params.
import main


# Main class
class Application:
    def __init__(self, master):
        # Write GUI only functions here:

        # Initialize title bar variables
        lastClickX = 0
        lastClickY = 0

        # Saves last click position for custom title bar.
        def SaveLastClickPos(event):
            global lastClickX, lastClickY
            lastClickX = event.x
            lastClickY = event.y

        # Allows dragging the title bar window.
        def Dragging(event):
            x, y = event.x - lastClickX + root.winfo_x(), event.y - lastClickY + root.winfo_y()
            root.geometry("+%s+%s" % (x, y))

        # Example internal function
        def gui_only_function():
            pass

        # Custom title bar
        title_bar = Frame(master, bg='#2e2e2e', relief='raised', bd=2, highlightthickness=0)
        # Close button for title bar
        close_application = Button(title_bar, text='X', command=master.destroy, bg="#2e2e2e", padx=2, pady=2,
                                   activebackground='red', bd=0, font="bold", fg='white', highlightthickness=0)

        # Favicon Icon - Initializes the asset.
        favicon = Image.open("assets/logo.jpg")
        favicon = favicon.resize((25, 25))

        # Converts the favicon asset to a Tkinter PhotoImage
        favicon_icon = ImageTk.PhotoImage(favicon)
        favicon_label = Label(title_bar, image=favicon_icon, borderwidth=0, highlightthickness=0)
        favicon_label.photo = favicon_icon
        favicon_label.pack(side=LEFT)

        # Title text for the label
        title_bar_text_label = Label(title_bar, text="Cough Classifier", bg="#2e2e2e", fg="#fff", font=("Hind", 10),
                                     padx=3)
        title_bar_text_label.pack(side=LEFT)

        title_bar.pack(expand=1, fill=X)
        close_application.pack(side=RIGHT)

        # Binds button 1 to two functions for clicking and dragging custom title bar.
        title_bar.bind('<Button-1>', SaveLastClickPos)
        title_bar.bind('<B1-Motion>', Dragging)

        # Creates master Canvas (not a frame)
        window = Canvas(master, width="800", height="600", bg='#2e2e2e', highlightthickness=0)
        window.pack(fill=BOTH)

        # Header frame
        header_frame = Frame(window, bg="#2e2e2e")
        window.create_window((0, 0), window=header_frame, anchor="nw", height=220, width=800)

        header_image = Image.open("assets/header.jpg")
        header_image = header_image.resize((180, 220))
        header_image_tk = ImageTk.PhotoImage(header_image)
        header_image_label = Label(header_frame, image=header_image_tk, borderwidth=0, highlightthickness=0)
        header_image_label.photo = header_image_tk
        header_image_label.pack(anchor='center')

        # Options Left frame
        options_left_frame = Frame(window, bg="#2e2e2e", relief='groove', bd=1, highlightthickness=0)
        window.create_window((0, 225), window=options_left_frame, anchor="nw", height=249, width=400)

        upload_image = Image.open("assets/upload_test_data.jpg")
        upload_image = upload_image.resize((225, 40))
        upload_image_tk = ImageTk.PhotoImage(upload_image)
        upload_image_label = Label(options_left_frame, image=upload_image_tk, borderwidth=0, highlightthickness=0)
        upload_image_label.photo = upload_image_tk
        upload_image_label.pack(anchor='center', pady=5)

        # Upload Test Cough Data
        def test_upload_cough_sound():
            location = "Sound_Folders/Test_Sounds/Cough_Test_Sounds/"
            filetypes = [("Cough Sounds", ".wav")]
            test_upload_cough_sound_assets = fd.askopenfilenames(parent=options_left_frame,
                                                                 title="Open Files", filetypes=filetypes)

            main.copy_uploaded_files_to_cough_test(test_upload_cough_sound_assets, location)

        # Upload Test Ambient Data
        def test_upload_ambient_sound():
            location = "Sound_Folders/Test_Sounds/Non_Cough_Test_Sounds/"
            filetypes = [("Ambient Sounds", ".wav")]
            test_upload_ambient_sound_assets = fd.askopenfilenames(parent=options_left_frame,
                                                                   title="Open Files", filetypes=filetypes)
            main.copy_uploaded_files_to_cough_test(test_upload_ambient_sound_assets, location)

        # Upload Classifier Cough Data
        def classifier_upload_cough_sound():
            location = "Sound_Folders/Cough_Recordings/"
            filetypes = [("Cough Sounds", ".wav")]
            classifier_upload_cough_sound_assets = fd.askopenfilenames(parent=options_left_frame,
                                                                       title="Open Files", filetypes=filetypes)
            main.copy_uploaded_files_to_cough_test(classifier_upload_cough_sound_assets, location)

        # Upload Classifier Ambient Data
        def classifier_upload_ambient_sound():
            location = "Sound_Folders/Training_Sounds/"
            filetypes = [("Ambient Sounds", ".wav")]
            classifier_upload_ambient_sound_assets = fd.askopenfilenames(parent=options_left_frame,
                                                                         title="Open Files", filetypes=filetypes)
            main.copy_uploaded_files_to_cough_test(classifier_upload_ambient_sound_assets, location)

        # Right/Left frame images

        image_cough = Image.open("assets/cough_button.jpg")
        image_cough = image_cough.resize((155, 45), Image.ANTIALIAS)
        self.reset_img_cough = ImageTk.PhotoImage(image_cough)

        image_ambient = Image.open("assets/ambient_button.jpg")
        image_ambient = image_ambient.resize((155, 45), Image.ANTIALIAS)
        self.reset_image_ambient = ImageTk.PhotoImage(image_ambient)

        # Buttons to upload test sound data
        test_upload_cough_sound_button = Button(options_left_frame, text="Upload Coughs", image=self.reset_img_cough,
                                                command=test_upload_cough_sound, bg='#05345C')
        test_upload_cough_sound_button.pack(anchor='center', pady=20)

        test_upload_ambient_sound_button = Button(options_left_frame, text="Upload Ambient",
                                                  image=self.reset_image_ambient, command=test_upload_ambient_sound,
                                                  bg='#05345C')
        test_upload_ambient_sound_button.pack(anchor='center', pady=20)

        # Options Right frame
        options_right_frame = Frame(window, bg="#2e2e2e", relief='groove', bd=1, highlightthickness=0)
        window.create_window((400, 225), window=options_right_frame, anchor="nw", height=249, width=401)

        options_image = Image.open("assets/upload_classifier_data.jpg")
        options_image = options_image.resize((225, 40))
        options_image_tk = ImageTk.PhotoImage(options_image)
        options_image_label = Label(options_right_frame, image=options_image_tk, borderwidth=0, highlightthickness=0)
        options_image_label.photo = options_image_tk
        options_image_label.pack(anchor='center', pady=5)

        # Buttons to upload Classifier sound data
        classifier_upload_cough_sound_button = Button(options_right_frame, text="Upload Coughs",
                                                      image=self.reset_img_cough, command=classifier_upload_cough_sound,
                                                      bg='#05345C')
        classifier_upload_cough_sound_button.pack(anchor='center', pady=20)

        classifier_upload_ambient_sound_button = Button(options_right_frame, text="Upload Ambient",
                                                        image=self.reset_image_ambient,
                                                        command=classifier_upload_ambient_sound, bg='#05345C')
        classifier_upload_ambient_sound_button.pack(anchor='center', pady=20)

        # Footer frame
        footer_frame = Frame(window, bg="#2e2e2e", relief='groove', bd=1, highlightthickness=0)
        window.create_window((0, 470), window=footer_frame, anchor="nw", height=100, width=801)  #

        # Button footer images
        image_execute = Image.open("assets/execute_test.jpg")
        image_execute = image_execute.resize((180, 52), Image.ANTIALIAS)
        self.reset_image_execute = ImageTk.PhotoImage(image_execute)

        image_train = Image.open("assets/train_model.jpg")
        image_train = image_train.resize((180, 52), Image.ANTIALIAS)
        self.reset_image_train = ImageTk.PhotoImage(image_train)

        image_record = Image.open("assets/record_audio.jpg")
        image_record = image_record.resize((180, 52), Image.ANTIALIAS)
        self.reset_image_record = ImageTk.PhotoImage(image_record)

        # Buttons for Footer

        run_test = Button(footer_frame, text="Execute Test", relief="flat",
                          image=self.reset_image_execute, command=main.test_function)
        run_test.pack(side=LEFT, padx=40)

        record_audio = Button(footer_frame, text="Record Audio", image=self.reset_image_record, relief="flat",
                              command=main.record_audio)
        record_audio.pack(side=LEFT, padx=40)

        train_data = Button(footer_frame, text="Train the Classifier", image=self.reset_image_train, relief="flat",
                            command=main.test_function)
        train_data.pack(side=LEFT, padx=40)


# Initialize GUI
root = Tk()
app = Application(root)
root.title('Cough Classifier')
# Sets windows size
root.geometry("800x600")
# Overrides tkinter styling (for title bar)
root.overrideredirect(True)
root.attributes('-topmost', True)
root.mainloop()
