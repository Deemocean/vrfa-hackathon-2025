import os
from dm_control import mjcf

def create_arena(xml_path_1, xml_path_2):
    print(f"Loading robots from {xml_path_1} and {xml_path_2}...")
    
    # 1. Load the models
    robot1 = mjcf.from_path(xml_path_1)
    robot2 = mjcf.from_path(xml_path_2)
    
    # 2. Create an empty arena
    arena = mjcf.RootElement(model="arena")
    
    # Add a floor
    chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                              height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                         texrepeat=[5, 5], reflectance=.2)
    arena.worldbody.add('geom', type='plane', size=[0, 0, .05], material=grid)
    
    # Add lights
    arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1], castshadow='true')

    # 3. Spawn robots with namespace prefixes
    # This prevents name collisions (e.g. "torso" vs "torso")
    # Attach robot 1
    spawn_site1 = arena.worldbody.add('site', pos=[-1, 0, 0])
    spawn_site1.attach(robot1)
    
    # Attach robot 2
    spawn_site2 = arena.worldbody.add('site', pos=[1, 0, 0])
    spawn_site2.attach(robot2)
    
    return arena

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_xml = os.path.join(base_dir, "assets", "humanoid", "humanoid.xml")
    output_dir = os.path.join(base_dir, "assets", "generated")
    output_xml_name = "arena.xml"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create arena with two of the same robot
    arena_model = create_arena(source_xml, source_xml)
    
    # Save
    print(f"Saving to {os.path.join(output_dir, output_xml_name)}...")
    mjcf.export_with_assets(arena_model, out_dir=output_dir, out_file_name=output_xml_name)
    print("Done! You can verify this by dragging the generated XML into the MuJoCo simulator.")
